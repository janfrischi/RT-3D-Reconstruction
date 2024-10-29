import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch

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

# Define class names for the detected objects
class_names = {0: "Person", 39: "Bottle", 41: "Cup", 62: "Laptop", 64: "Mouse", 66: "Keyboard"}

def erode_mask(mask, iterations=1):
    kernel = np.ones((8, 8), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    mask_indices = torch.tensor(mask_indices, device=device)
    # Extract the u and v coordinates from the mask indices
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    depth_values = depth_map[v_coords, u_coords].to(device)
    valid_mask = depth_values > 0
    # Extract the valid u, v, and depth values
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depth_values = depth_values[valid_mask]
    # Calculate the x, y, and z coordinates of the 3D points using the camera intrinsics
    # Where fx,fy are the focal lengths and cx,cy are the principal points
    x_coords = (u_coords - cx) * depth_values / fx
    y_coords = (v_coords - cy) * depth_values / fy
    z_coords = depth_values
    # Return a tensor of shape (N, 3) representing the 3D points, stacked along the last dimension
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

    # Transformation matrices from the chessboard to the camera frames

    T_chess_cam1 = np.array([[-0.9770, 0.1087, -0.1835, 0.3060],
                             [-0.2100, -0.6401, 0.7390, -0.7103],
                             [-0.0371, 0.7605, 0.6482, -0.6424],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[0.6760, 0.4811, -0.5582, 0.6475],
                             [-0.7369, 0.4343, -0.5180, 0.8451],
                             [-0.0068, 0.7615, 0.6481, -0.7279],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    # Transformation matrix from the chessboard to the robot base frame

    T_robot_chess = np.array([[-1, 0, 0, 0.3580],
                              [0, 1, 0, 0.0300],
                              [0, 0, -1, 0.0060],
                              [0, 0, 0, 1]])

    # Transformation from camera frame (left camera of respective stereo camera) to robot base frame
    T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
    T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)

    # Create a robot base frame
    robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    robot_base_frame.transform(T_robot_cam1)

    # Create Mat object to hold the image and depth map
    image = sl.Mat()
    depth = sl.Mat()
    key = ''
    print("Press 'q' to quit the video feed.")

    # Initialize Open3D visualizer (using Visualizer for full interaction)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Visualization-Pointcloud', width=img_width, height=img_height)
    vis.add_geometry(coordinate_frame)

    # Create a named window for the video feed
    cv2.namedWindow("YOLO11 Segmentation+Tracking")

    fps_values = []
    frame_count = 0
    update_frequency = 5  # Update every 5 frames -> Significant performance improvement in visualization

    # Real-Time Loop for Capturing and Processing Frames
    while key != ord('q'):
        start_time = time.time()

        # Image and Depth Capture
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            depth_retrieval_result = zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            if depth_retrieval_result != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result}")
                continue

            frame = image.get_data()

            # Convert frame from 4 channels (RGBA) to 3 channels (BGR) -> OpenCV uses BGR format
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

            # Visualize the results on the frame using the plot method
            annotated_frame = results[0].plot(line_width=2, font_size=18)

            # Get the mask and class IDs for the detected objects
            masks = results[0].masks
            # Convert the class IDs to a NumPy array and move it to the CPU
            class_ids = results[0].boxes.cls.cpu().numpy()
            # List to store the 3D point clouds of the detected objects
            point_clouds = []

            # Process each mask -> masks.data is a tensor of shape (N, H, W) where N is the number of masks
            for i, mask in enumerate(masks.data):
                # Convert the mask tensor to a NumPy array and move it to the CPU
                mask = mask.cpu().numpy()
                mask = erode_mask(mask, iterations=1)
                # Get the indices of the non-zero elements in the mask
                mask_indices = np.argwhere(mask > 0)

                # Convert the ZED depth map to a PyTorch tensor and move it to the GPU
                depth_map = torch.from_numpy(zed_depth_np).to(device)

                # Calculate 3D points from the mask indices and depth map
                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy) # Returns a tensor of shape (N, 3) "List of 3D points"

                # Check if any 3D points are returned
                if points_3d.size(0) > 0:
                    point_cloud_np = points_3d.cpu().numpy()
                    point_clouds.append(point_cloud_np)
                    class_id = int(class_ids[i])
                    print(f"Class ID: {class_id} ({class_names[class_id]}) in Camera Frame 1")

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

                    # TODO: Check if the transformation matrix is correct
                    # Transformation matrix for 180° rotation around the x-axis
                    transformation_matrix = np.array([[1, 0, 0, 0],
                                                      [0, -1, 0, 0],
                                                      [0, 0, -1, 0],
                                                      [0, 0, 0, 1]])
                    pcd.transform(transformation_matrix)

                    vis.add_geometry(pcd)

                vis.add_geometry(coordinate_frame)
                vis.add_geometry(robot_base_frame)
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

    # If q is pressed the loop will break and the camera will be closed
    zed.close()
    vis.destroy_window()

if __name__ == "__main__":
    main()