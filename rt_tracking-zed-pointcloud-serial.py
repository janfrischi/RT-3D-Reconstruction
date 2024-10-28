import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
from shapely.constructive import centroid

from ultralytics import YOLO
import open3d as o3d
import random

# Define a color map for different classes
color_map = {
    0: [15, 82, 186],  # Person - sapphire
    39: [255, 255, 0],  # Bottle - yellow
    41: [63, 224, 208],  # Cup - turquoise
    62: [255, 0, 255],  # Laptop - magenta
    64: [0, 0, 128],  # Mouse - navy
    66: [255, 0, 0]  # Keyboard - red
}

def erode_mask(mask, iterations=1):
    kernel = np.ones((8, 8), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    mask_indices = torch.tensor(mask_indices, device=device)
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    depth_values = depth_map[v_coords, u_coords].to(device)
    valid_mask = depth_values > 0
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depth_values = depth_values[valid_mask]
    x_coords = (u_coords - cx) * depth_values / fx
    y_coords = (v_coords - cy) * depth_values / fy
    z_coords = depth_values
    return torch.stack((x_coords, y_coords, z_coords), dim=-1)

def random_sample_pointcloud(pc, fraction):
    """
    Randomly sample a fraction of the points in a point cloud.
    Args:
        pc: Input point cloud (Nx3).
        fraction: Fraction of points to retain.
    Returns:
        Sampled point cloud.
    """
    n_points = pc.shape[0]
    sample_size = int(n_points * fraction)
    if sample_size <= 0:
        return pc
    indices = random.sample(range(n_points), sample_size)
    return pc[indices]

def calculate_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)

def main():

    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    transformation_matrix = np.array([[1, 0, 0, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0, 0, 1]])
    coordinate_frame.transform(transformation_matrix)
    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pretrained YOLO11 segmentation model and move it to the device
    model = YOLO("yolo11m-seg.pt").to(device)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = 0.3
    init_params.coordinate_units = sl.UNIT.METER

    # Check if the camera is opened successfully
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        exit(1)

    # Get the camera intrinsics for the left camera
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy
    cx, cy = calibration_params.left_cam.cx, calibration_params.left_cam.cy
    img_width, img_height = calibration_params.left_cam.image_size.width, calibration_params.left_cam.image_size.height

    # Define camera parameters
    lookat = [0.0, 0.0, 0.0]  # Camera is looking at the origin
    front = [0.0, 0.0, -1.0]  # Camera is facing along the negative z-axis
    up = [0, 1.0, 0]  # The upward direction is along the y-axis
    zoom = 1  # Adjust zoom to zoom in slightly

    # Create Mat object to hold the image and depth map
    image = sl.Mat()
    depth = sl.Mat()
    key = ''
    print("Press 'q' to quit the video feed.")

    # Initialize Open3D visualizer (using Visualizer for full interaction)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Visualization-Pointcloud', width=img_width, height=img_height)
    vis.add_geometry(coordinate_frame)

    cv2.namedWindow("YOLO11 Segmentation+Tracking")

    fps_values = []
    frame_count = 0
    update_frequency = 5  # Update every 5 frames -> Significant performance improvement in visualization

    # Main loop responsible for capturing and processing frames
    while key != ord('q'):
        start_time = time.time()

        # Grab an image from the camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Check if depth retrieval is successful
            depth_retrieval_result = zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            if depth_retrieval_result != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result}")
                continue

            frame = image.get_data()

            # Convert frame from 4 channels (RGBA) to 3 channels (BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLO11 inference on the frame
            results = model.track(
                source=frame,
                imgsz=640,
                max_det=20,
                classes=[0, 39, 41, 62, 64, 66],
                half=True,
                persist=True,
                retina_masks=True,
                conf=0.5,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            # Convert the ZED depth map to a NumPy array
            zed_depth_np = depth.get_data()

            if zed_depth_np is None:
                print("Error: Depth map is empty")
                continue

            # Visualize the results on the frame
            annotated_frame = results[0].plot(line_width=2, font_size=18)

            # Get the mask and class IDs for the detected objects
            masks = results[0].masks
            class_ids = results[0].boxes.cls.cpu().numpy()
            point_clouds = []

            # Process each mask
            for i, mask in enumerate(masks.data):
                mask = mask.cpu().numpy()
                mask = erode_mask(mask, iterations=1)
                mask_indices = np.argwhere(mask > 0)

                depth_map = torch.from_numpy(zed_depth_np).to(device)

                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy)

                if points_3d.size(0) > 0:
                    point_cloud_np = points_3d.cpu().numpy()
                    point_clouds.append(point_cloud_np)

                    # Calculate the centroid of the point cloud
                    centroid = calculate_centroid(point_cloud_np)
                    class_id = int(class_ids[i])
                    print(f"Class ID: {class_id}, CentroidXYZ-Coords: {centroid} in Camera Frame 1")


            # Visualize the 3D point clouds every 'update_frequency' frames
            if point_clouds and frame_count % update_frequency == 0:
                vis.clear_geometries()

                for i, pc in enumerate(point_clouds):
                    # Randomly sample a fraction of the points
                    sampled_pc = random_sample_pointcloud(pc, fraction=0.05)  # Retain 5% of points
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_pc))

                    class_id = int(class_ids[i])
                    color = np.array(color_map.get(class_id, [1, 1, 1])) / 255.0
                    pcd.paint_uniform_color(color)

                    # Transformation matrix for 180° rotation around the x-axis
                    transformation_matrix = np.array([[1, 0, 0, 0],
                                                      [0, -1, 0, 0],
                                                      [0, 0, -1, 0],
                                                      [0, 0, 0, 1]])
                    pcd.transform(transformation_matrix)

                    vis.add_geometry(pcd)

                vis.add_geometry(coordinate_frame)
                vis.poll_events()
                vis.update_renderer()

            # Capture static scene if 's' is pressed
            if key == ord('s') and point_clouds:
                print("Capturing static image of the scene")
                captured_point_clouds = []
                for i, pc in enumerate(point_clouds):
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
                    pcd.transform(transformation_matrix)
                    class_id = int(class_ids[i])
                    color = np.array(color_map.get(class_id, [1, 1, 1])) / 255.0
                    pcd.paint_uniform_color(color)
                    captured_point_clouds.append(pcd)
                captured_point_clouds.append(coordinate_frame)
                o3d.visualization.draw_geometries(captured_point_clouds)


            frame_count += 1

            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_values.append(fps)

            # Calculate a moving average to smooth the FPS display
            if len(fps_values) > 10:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            # Display FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize the window to match the frame resolution
            height, width, _ = frame.shape
            cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
            cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame)

            key = cv2.waitKey(1)

    # Close the camera and exit
    zed.close()
    vis.destroy_window()

if __name__ == "__main__":
    main()
