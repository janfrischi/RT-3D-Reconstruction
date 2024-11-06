import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import random

from ultralytics import YOLO

# Define a color map for different classes
color_map = {
    0: [15, 82, 186],  # Person - sapphire
    39: [255, 255, 0],  # Bottle - yellow
    41: [63, 224, 208],  # Cup - turquoise
    62: [255, 0, 255],  # Laptop - magenta
    64: [0, 0, 128],  # Mouse - navy
    66: [255, 0, 0],  # Keyboard - red
    73: [0, 255, 0]   # Book - green
}

# Define class names for the detected objects
class_names = {0: "Person",
               39: "Bottle",
               41: "Cup",
               62: "Laptop",
               64: "Mouse",
               66: "Keyboard",
               73: "Book"}

# Erode the mask to remove background noise, since masks aren't perfect
def erode_mask(mask, iterations=1):
    kernel = np.ones((14, 14), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    mask_indices = torch.tensor(mask_indices, device=device)
    # Extract the u, v coordinates from the mask indices
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    # Extract the depth values from the depth map and move to GPU
    depth_values = depth_map[v_coords, u_coords].to(device)
    valid_mask = depth_values > 0
    # Filter out invalid depth values
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depth_values = depth_values[valid_mask]
    # Calculate the 3D coordinates using the camera intrinsics
    x_coords = (u_coords - cx) * depth_values / fx
    y_coords = (v_coords - cy) * depth_values / fy
    z_coords = depth_values
    # Return a tensor of shape (N, 3) containing the 3D coordinates
    return torch.stack((x_coords, y_coords, z_coords), dim=-1)


# Sample a random fraction of points from a point cloud. Function returns the sampled point cloud
def random_sample_pointcloud(pc, fraction):
    n_points = pc.shape[0]
    sample_size = int(n_points * fraction)
    if sample_size <= 0:
        return pc
    indices = random.sample(range(n_points), sample_size)
    return pc[indices]

def main():
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pre-trained YOLOv11 model and move it to the device
    model = YOLO("yolo11l-seg.pt").to(device)

    # Initialize the ZED camera objects
    zed1 = sl.Camera()
    zed2 = sl.Camera()

    # Set the serial numbers of the cameras
    sn_cam1 = 33137761
    sn_cam2 = 30635524

    # Set the initialization parameters for camera 1
    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 60
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_units = sl.UNIT.METER

    # Set the initialization parameters for camera 2
    init_params2 = sl.InitParameters()
    init_params2.set_from_serial_number(sn_cam2)
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 60
    init_params2.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params2.depth_minimum_distance = 0.4
    init_params2.coordinate_units = sl.UNIT.METER

    # Check if the cameras were successfully opened
    err1 = zed1.open(init_params1)
    if err1 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 1: {err1}")
        exit(1)

    err2 = zed2.open(init_params2)
    if err2 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 2: {err2}")
        exit(1)

    # Get the camera calibration parameters
    calibration_params1 = zed1.get_camera_information().camera_configuration.calibration_parameters
    fx1, fy1 = calibration_params1.left_cam.fx, calibration_params1.left_cam.fy
    cx1, cy1 = calibration_params1.left_cam.cx, calibration_params1.left_cam.cy

    calibration_params2 = zed2.get_camera_information().camera_configuration.calibration_parameters
    fx2, fy2 = calibration_params2.left_cam.fx, calibration_params2.left_cam.fy
    cx2, cy2 = calibration_params2.left_cam.cx, calibration_params2.left_cam.cy

    # Define the transformation matrices from the chessboard to the camera frames
    T_chess_cam1 = np.array([[0.6629, 0.4872, -0.5685, 0.5789],
                             [-0.7487, 0.4262, -0.5077, 0.7758],
                             [-0.0050, 0.7622, 0.6473, -0.7253],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[0.3281, -0.6660, 0.6699, -0.5230],
                             [0.9445, 0.2437, -0.2204, 0.3022],
                             [-0.0165, 0.7051, 0.7089, -0.6026],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_robot_chess = np.array([[-1, 0, 0, 0.3580],
                              [0, 1, 0, 0.0300],
                              [0, 0, -1, 0.0060],
                              [0, 0, 0, 1]])

    # Calculate the transformation matrices from the robot frame to the camera frames
    T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
    T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)

    # Extract the rotation matrices and translation vectors from the transformation matrices
    rotation_robot_cam1 = T_robot_cam1[:3, :3]
    rotation_robot_cam2 = T_robot_cam2[:3, :3]

    origin_cam1 = T_robot_cam1[:3, 3]
    origin_cam2 = T_robot_cam2[:3, 3]

    print(f"Rotation matrix from robot frame to camera frame 1:\n{rotation_robot_cam1}")
    print(f"Rotation matrix from robot frame to camera frame 2:\n{rotation_robot_cam2}")
    print(f"Translation vector from robot frame to camera frame 1: {origin_cam1}")
    print(f"Translation vector from robot frame to camera frame 2: {origin_cam2}")

    distance_cam1 = np.linalg.norm(origin_cam1 - np.array([0, 0, 0]))
    distance_cam2 = np.linalg.norm(origin_cam2 - np.array([0, 0, 0]))

    print(f"Distance from robot frame to camera frame 1: {distance_cam1:.4f} meters")
    print(f"Distance from robot frame to camera frame 2: {distance_cam2:.4f} meters")

    # Initialize the image and depth map variables for both cameras
    image1 = sl.Mat()
    depth1 = sl.Mat()
    image2 = sl.Mat()
    depth2 = sl.Mat()
    key = ''

    # Create a window to display the output
    cv2.namedWindow("YOLO11 Segmentation+Tracking")
    fps_values = []
    frame_count = 0

    # Initialize lists to store the point clouds from both cameras, each point cloud is a tuple of the point cloud and the class ID
    point_clouds_camera1 = []
    point_clouds_camera2 = []

    # Main loop to capture and process images from both cameras
    while key != ord('q'):
        start_time = time.time()

        if zed1.grab() == sl.ERROR_CODE.SUCCESS and zed2.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the images and depth maps from both cameras
            zed1.retrieve_image(image1, sl.VIEW.LEFT)
            zed2.retrieve_image(image2, sl.VIEW.LEFT)
            depth_retrieval_result1 = zed1.retrieve_measure(depth1, sl.MEASURE.DEPTH)
            depth_retrieval_result2 = zed2.retrieve_measure(depth2, sl.MEASURE.DEPTH)

            # Check if the depth maps were successfully retrieved
            if depth_retrieval_result1 != sl.ERROR_CODE.SUCCESS or depth_retrieval_result2 != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result1}, {depth_retrieval_result2}")
                continue

            frame1 = image1.get_data()
            frame2 = image2.get_data()
            # Convert the frames from RGBA to RGB as this is the format expected by OPENCV
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)

            # Perform object detection and tracking on both frames
            results1 = model.track(
                source=frame1,
                imgsz=640,
                vid_stride = 5,
                classes=[39, 41, 62, 64, 66, 73],
                half=True,
                persist=True,
                retina_masks=True,
                conf=0.2,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            results2 = model.track(
                source=frame2,
                imgsz=640,
                vid_stride=5,
                classes=[39, 41, 62, 64, 66, 73],
                half=True,
                persist=True,
                retina_masks=True,
                conf=0.2,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            # Retrieve the depth maps from both cameras
            zed_depth_np1 = depth1.get_data()
            zed_depth_np2 = depth2.get_data()

            # Check if the depth maps are empty
            if zed_depth_np1 is None or zed_depth_np2 is None:
                print("Error: Depth map is empty")
                continue

            # Plot the annotated frames with bounding boxes and class labels
            annotated_frame1 = results1[0].plot(line_width=2, font_size=18)
            annotated_frame2 = results2[0].plot(line_width=2, font_size=18)

            # Retrieve the masks and class IDs from the results of the object detection
            masks1 = results1[0].masks
            masks2 = results2[0].masks
            class_ids1 = results1[0].boxes.cls.cpu().numpy()
            class_ids2 = results2[0].boxes.cls.cpu().numpy()

            # Get depth-maps from both cameras
            depth_map1 = torch.from_numpy(zed_depth_np1).to(device)
            depth_map2 = torch.from_numpy(zed_depth_np2).to(device)

            # Iterate over the masks and class IDs to extract the 3D points for each detected object
            for i, mask in enumerate(masks1.data):
                mask = mask.cpu().numpy()
                mask = erode_mask(mask, iterations=1)
                # Move the mask indices to the GPU
                mask_indices = np.argwhere(mask > 0)

                # Calculate the 3D points using the mask indices and depth map -> This operation is done on the GPU
                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map1, cx1, cy1, fx1, fy1)

                if points_3d.size(0) > 0:
                    # Move the 3D points to the CPU and transform them to the robot base frame
                    point_cloud_cam1 = points_3d.cpu().numpy()
                    point_cloud_np_transformed = np.dot(rotation_robot_cam1, point_cloud_cam1.T).T + origin_cam1
                    # Add the point cloud and class ID to the list
                    point_clouds_camera1.append((point_cloud_np_transformed, int(class_ids1[i])))
                    print(f"Class ID: {class_ids1[i]} ({class_names[class_ids1[i]]}) in Camera Frame 1")

            for i, mask in enumerate(masks2.data):
                mask = mask.cpu().numpy()
                mask = erode_mask(mask, iterations=1)
                mask_indices = np.argwhere(mask > 0)

                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map2, cx2, cy2, fx2, fy2)

                if points_3d.size(0) > 0:
                    point_cloud_cam2 = points_3d.cpu().numpy()
                    point_cloud_np_transformed = np.dot(rotation_robot_cam2, point_cloud_cam2.T).T + origin_cam2
                    point_clouds_camera2.append((point_cloud_np_transformed, int(class_ids2[i])))
                    print(f"Class ID: {class_ids2[i]} ({class_names[class_ids2[i]]}) in Camera Frame 2")

            # Increment the frame count and calculate the FPS
            frame_count += 1
            fps = 1.0 / (time.time() - start_time)
            fps_values.append(fps)

            if len(fps_values) > 10:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            # Display the annotated frames with the FPS
            if annotated_frame1 is not None:
                cv2.putText(annotated_frame1, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                height, width, _ = frame1.shape
                cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
                cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame1)

            if annotated_frame2 is not None:
                cv2.putText(annotated_frame2, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                height, width, _ = frame2.shape
                cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
                cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame2)

            # Concatenate the frames horizontally
            combined_frame = cv2.hconcat([annotated_frame1, annotated_frame2])
            cv2.imshow("YOLO11 Segmentation+Tracking", combined_frame)

            key = cv2.waitKey(1)

    zed1.close()
    zed2.close()

    # Access the point clouds from both cameras
    print("Point clouds from Camera 1:")
    for pc, class_id in point_clouds_camera1:
        print(f"Class ID: {class_id}, Point Cloud Shape: {pc.shape}")

    print("Point clouds from Camera 2:")
    for pc, class_id in point_clouds_camera2:
        print(f"Class ID: {class_id}, Point Cloud Shape: {pc.shape}")

if __name__ == "__main__":
    main()