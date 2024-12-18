import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import csv
from ultralytics import YOLO
from vision_pipeline_utils import convert_mask_to_3d_points, downsample_point_cloud_gpu, crop_point_cloud_gpu, fuse_point_clouds_centroid, subtract_point_clouds_gpu

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

# Dictionary to store cumulative timings for Benchmarking
timings = {
    "Frame Retrieval": [],
    "Depth Retrieval": [],
    "Point Cloud Processing": [],
    "YOLO11 Inference": [],
    "Mask Processing": [],
    "Point Cloud Fusion": [],
    "Subtraction": [],
    "Total Time per Iteration": []
}


def main():

    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pre-trained YOLOv11 model and move it to the device
    model = YOLO("models/yolo11x-seg.pt").to(device)

    # Initialize the CSV file to store the results
    fps_log_file = "fps_log.csv"

    # Create or overwrite the CSV file and write the header
    with open(fps_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "FPS"])

    # Initialize the ZED camera objects
    zed1 = sl.Camera()
    zed2 = sl.Camera()

    # Set the serial numbers of the cameras
    sn_cam1 = 33137761
    sn_cam2 = 36829049

    # Set the initialization parameters for camera 1
    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_units = sl.UNIT.METER

    # Set the initialization parameters for camera 2
    init_params2 = sl.InitParameters()
    init_params2.set_from_serial_number(sn_cam2)
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 30
    init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
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

    T_chess_cam1 = np.array([[0.8401, 0.3007, -0.4515, 0.5914],
                             [-0.5424, 0.4647, -0.6999, 1.1329],
                             [-0.0006, 0.8329, 0.5535, -0.7193],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[0.4608, -0.4105, 0.7869, -0.8716],
                             [0.8874, 0.2010, -0.4148, 0.8318],
                             [0.0121, 0.8895, 0.4569, -0.7200],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    # This transformation matrix is given by the geometry of the mount and the chessboard
    T_robot_chess = np.array([[-1.0000, 0.0000, 0.0000, 0.3580],
                              [0.0000, 1.0000, 0.0000, 0.0300],
                              [0.0000, 0.0000, -1.0000, 0.0060],
                              [0.0000, 0.0000, 0.0000, 1.0000]])

    # Calculate the transformation matrices from the robot frame to the camera frames
    T_robot_cam1 = np.dot(T_robot_chess, T_chess_cam1)
    T_robot_cam2 = np.dot(T_robot_chess, T_chess_cam2)

    # Extract the rotation matrices and translation vectors from the transformation matrices
    rotation_robot_cam1 = T_robot_cam1[:3, :3]
    rotation_robot_cam2 = T_robot_cam2[:3, :3]
    origin_cam1 = T_robot_cam1[:3, 3]
    origin_cam2 = T_robot_cam2[:3, 3]

    # Convert them to tensors
    rotation1_torch = torch.tensor(rotation_robot_cam1, dtype=torch.float32, device=device)
    origin1_torch = torch.tensor(origin_cam1, dtype=torch.float32, device=device)
    rotation2_torch = torch.tensor(rotation_robot_cam2, dtype=torch.float32, device=device)
    origin2_torch = torch.tensor(origin_cam2, dtype=torch.float32, device=device)

    # Calculate the distance from the robot frame to the camera frames
    distance_cam1 = np.linalg.norm(origin_cam1 - np.array([0, 0, 0]))
    distance_cam2 = np.linalg.norm(origin_cam2 - np.array([0, 0, 0]))

    print(f"Distance from robot frame to camera frame 1: {distance_cam1:.4f} meters")
    print(f"Distance from robot frame to camera frame 2: {distance_cam2:.4f} meters")

    # Set the resolution for the point clouds
    #resolution = sl.Resolution(1280, 720)
    # Using a lower resolution for faster processing
    resolution = sl.Resolution(640, 360)

    # Initialize the image and depth map objects for both cameras
    image1 = sl.Mat()
    depth1 = sl.Mat()
    image2 = sl.Mat()
    depth2 = sl.Mat()

    # Create point cloud objects to hold data of the workspace
    point_cloud1_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud2_ws = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # Initialize the key variable to check for the 'q' key press
    key = ''

    # Create a window to display the output
    cv2.namedWindow("YOLO11 Segmentation+Tracking")
    fps_values = []
    frame_count = 0

    # Initialize lists to store the point clouds of the reconstructed objects from both cameras, each point cloud is a tuple of the point cloud and the class ID
    point_clouds_camera1 = []
    point_clouds_camera2 = []

    # Main loop to capture and process images from both cameras
    while key != ord('q'):
        start_time = time.time()
        # Step 1: Frame retrieval
        if zed1.grab() == sl.ERROR_CODE.SUCCESS and zed2.grab() == sl.ERROR_CODE.SUCCESS:
            retrieval_start_time = time.time()
            # Retrieve the images from both cameras and convert them to numpy arrays
            zed1.retrieve_image(image1, view=sl.VIEW.LEFT)
            zed2.retrieve_image(image2, view=sl.VIEW.LEFT)
            frame1 = image1.get_data()
            frame2 = image2.get_data()
            # Convert the frames from RGBA to RGB as this is the format expected by OPENCV
            # frame1 and frame2 are the inputs to the YOLO model
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)
            retrieval_end_time = time.time()
            timings["Frame Retrieval"].append(retrieval_end_time - retrieval_start_time)
            print(f"Frame retrieval time: {retrieval_end_time - retrieval_start_time:.4f} seconds")

            # Step 2: Depth map retrieval
            # Retrieve the depth maps from both cameras and convert them to numpy arrays
            depth_start_time = time.time()
            depth_retrieval_result1 = zed1.retrieve_measure(depth1, measure=sl.MEASURE.DEPTH)
            depth_retrieval_result2 = zed2.retrieve_measure(depth2, measure=sl.MEASURE.DEPTH)
            # Convert the depth maps to numpy arrays
            zed_depth_np1 = depth1.get_data()
            zed_depth_np2 = depth2.get_data()
            depth_end_time = time.time()
            timings["Depth Retrieval"].append(depth_end_time - depth_start_time)
            print(f"Depth retrieval time: {depth_end_time - depth_start_time:.4f} seconds")

            # Check if the depth maps were successfully retrieved
            if depth_retrieval_result1 != sl.ERROR_CODE.SUCCESS or depth_retrieval_result2 != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result1}, {depth_retrieval_result2}")
                continue

            # Step 3: Point cloud retrieval and processing
            point_cloud_start_time = time.time()
            # 1. Retrieve the point clouds for the workspace from both cameras
            zed1.retrieve_measure(point_cloud1_ws, measure=sl.MEASURE.XYZ, resolution=resolution)
            zed2.retrieve_measure(point_cloud2_ws, measure=sl.MEASURE.XYZ, resolution=resolution)

            # 2. Convert the point clouds to tensors directly
            point_cloud1_ws_tensor = torch.tensor(point_cloud1_ws.get_data()[:, :, :3], dtype=torch.float32, device=device).reshape(-1, 3)
            point_cloud2_ws_tensor = torch.tensor(point_cloud2_ws.get_data()[:, :, :3], dtype=torch.float32, device=device).reshape(-1, 3)

            # 3. Filter invalid points from the point clouds using tensor operations
            valid_mask_workspace1 = torch.isfinite(point_cloud1_ws_tensor).all(dim=1)
            point_cloud1_ws_tensor = point_cloud1_ws_tensor[valid_mask_workspace1]
            valid_mask_workspace2 = torch.isfinite(point_cloud2_ws_tensor).all(dim=1)
            point_cloud2_ws_tensor = point_cloud2_ws_tensor[valid_mask_workspace2]

            # Transform the point clouds to the robot base frame using torch tensors for GPU acceleration
            point_cloud1_ws_transformed_tensor = torch.mm(rotation1_torch, point_cloud1_ws_tensor.T).T + origin1_torch
            point_cloud2_ws_transformed_tensor = torch.mm(rotation2_torch, point_cloud2_ws_tensor.T).T + origin2_torch

            # Crop the point clouds to the workspace
            x_bounds_baseframe = (-0.25, 0.75)
            y_bounds_baseframe = (-0.5, 1.75)
            z_bounds_baseframe = (-0.05, 2)

            point_cloud1_workspace_np_cropped = crop_point_cloud_gpu(
                point_cloud1_ws_transformed_tensor,
                x_bounds_baseframe,
                y_bounds_baseframe,
                z_bounds_baseframe
            )

            point_cloud2_workspace_np_cropped = crop_point_cloud_gpu(
                point_cloud2_ws_transformed_tensor,
                x_bounds_baseframe,
                y_bounds_baseframe,
                z_bounds_baseframe
            )

            # Downsample the point clouds
            point_cloud1_workspace = downsample_point_cloud_gpu(point_cloud1_workspace_np_cropped, voxel_size=0.005)
            point_cloud2_workspace = downsample_point_cloud_gpu(point_cloud2_workspace_np_cropped, voxel_size=0.005)
            fused_point_cloud_ws = torch.cat((point_cloud1_workspace, point_cloud2_workspace), dim=0)

            # SOR filter the fused point cloud
            #fused_point_cloud_ws = filter_outliers_sor_gpu(fused_point_cloud_ws, nb_neighbors=20, std_ratio=1.5)

            # Convert to numpy array
            fused_point_cloud_ws = fused_point_cloud_ws.cpu().numpy()

            # SOR Filtering on the CPU
            #fused_point_cloud_ws = filter_outliers_sor(fused_point_cloud_ws, nb_neighbors=20, std_ratio=1.5)

            print(f"Fused Point Cloud Workspace shape: {fused_point_cloud_ws.shape}")
            point_cloud_end_time = time.time()
            timings["Point Cloud Processing"].append(point_cloud_end_time - point_cloud_start_time)
            print(f"Point cloud processing time: {point_cloud_end_time - point_cloud_start_time:.4f} seconds")


            # Perform object detection/segmentation and tracking on both frames using YOLO11
            yolo11_start_time = time.time()
            yolo11_results1 = model.track(
                source=frame1,
                classes=[39, 41],
                persist=True,
                retina_masks=True,
                conf=0.1,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            yolo11_results2 = model.track(
                source=frame2,
                imgsz=640,
                classes=[39, 41],
                persist=True,
                retina_masks=True,
                conf=0.1,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            # Plot the annotated frames with bounding boxes and class labels and confidence scores
            annotated_frame1 = yolo11_results1[0].plot(line_width=2, font_size=18)
            annotated_frame2 = yolo11_results2[0].plot(line_width=2, font_size=18)

            # YOLO Outputs Retrieve the masks and class IDs from the results of the object detection
            masks1 = yolo11_results1[0].masks
            masks2 = yolo11_results2[0].masks
            class_ids1 = yolo11_results1[0].boxes.cls.cpu().numpy()
            class_ids2 = yolo11_results2[0].boxes.cls.cpu().numpy()
            yolo11_end_time = time.time()
            timings["YOLO11 Inference"].append(yolo11_end_time - yolo11_start_time)
            print(f"YOLO11 processing time: {yolo11_end_time - yolo11_start_time:.4f} seconds")

            start_time_processing_masks = time.time()
            # Processing the masks from camera 1
            if masks1 is not None:
                # Get depth-maps from both cameras, convert input Numpy arrays to PyTorch tensors and move them to the GPU
                depth_map1 = torch.tensor(zed_depth_np1, dtype=torch.float32, device=device)
                # Iterate over the masks and class IDs to extract the 3D points for each detected object
                for i, mask in enumerate(masks1.data):
                    #mask = erode_mask_gpu(mask, kernel_size=1)
                    mask_indices = torch.nonzero(mask, as_tuple=False)

                    # Calculate the 3D points using the mask indices and depth map -> This operation is done on the GPU,
                    # Points are stored as a tensor on the GPU, data type is torch.float32
                    with torch.amp.autocast('cuda'):
                        points_3d_cam1 = convert_mask_to_3d_points(mask_indices, depth_map1, cx1, cy1, fx1, fy1)

                    if points_3d_cam1.size(0) > 0:
                        # Transform points using torch.mm for GPU acceleration
                        rotation_robot_cam1_torch = torch.tensor(rotation_robot_cam1, dtype=torch.float32,
                                                                 device=points_3d_cam1.device)
                        origin_cam1_torch = torch.tensor(origin_cam1, dtype=torch.float32, device=points_3d_cam1.device)

                        # Perform transformation on the GPU
                        point_cloud_cam1_transformed = torch.mm(points_3d_cam1,
                                                                rotation_robot_cam1_torch.T) + origin_cam1_torch

                        # Downsample the point cloud of camera 1 on the GPU
                        point_cloud_cam1_downsampled = downsample_point_cloud_gpu(point_cloud_cam1_transformed,voxel_size=0.005)

                        # Move transformed points to CPU for further processing
                        point_cloud_cam1_downsampled_cpu = point_cloud_cam1_downsampled.cpu().numpy()

                        # Add the down sampled point cloud and class ID to this camera's point cloud list
                        point_clouds_camera1.append((point_cloud_cam1_downsampled_cpu, int(class_ids1[i])))


            # Processing the masks from camera 2
            if masks2 is not None:
                # Get depth-maps from both cameras
                depth_map2 = torch.tensor(zed_depth_np2, dtype=torch.float32, device=device)
                # Iterate over the masks and class IDs to extract the 3D points for each detected object
                for i, mask in enumerate(masks2.data):
                    #mask = erode_mask_gpu(mask, kernel_size=1)
                    mask_indices = torch.nonzero(mask, as_tuple=False)

                    with torch.amp.autocast('cuda'):
                        points_3d_cam2 = convert_mask_to_3d_points(mask_indices, depth_map2, cx2, cy2, fx2, fy2)

                    if points_3d_cam2.size(0) > 0:
                        # Transform points using torch.mm for GPU acceleration
                        rotation_robot_cam2_torch = torch.tensor(rotation_robot_cam2, dtype=torch.float32,
                                                                 device=points_3d_cam2.device)
                        origin_cam2_torch = torch.tensor(origin_cam2, dtype=torch.float32, device=points_3d_cam2.device)

                        # Perform transformation on the GPU
                        point_cloud_cam2_transformed = torch.mm(points_3d_cam2,
                                                                rotation_robot_cam2_torch.T) + origin_cam2_torch

                        # Downsample the point cloud of camera 2 on the GPU
                        point_cloud_cam2_downsampled = downsample_point_cloud_gpu(point_cloud_cam2_transformed,
                                                                                  voxel_size=0.005)
                        # Move transformed points to CPU for further processing
                        point_cloud_cam2_downsampled_cpu = point_cloud_cam2_downsampled.cpu().numpy()

                        # Add the downsampled point cloud and class ID to this camera's point cloud list
                        point_clouds_camera2.append((point_cloud_cam2_downsampled_cpu, int(class_ids2[i])))
                        print(f"Class ID: {class_ids2[i]} ({class_names[class_ids2[i]]}) in Camera Frame 2")

            end_time_processing_masks = time.time()
            timings["Mask Processing"].append(end_time_processing_masks - start_time_processing_masks)
            print(f"Processing masks time: {end_time_processing_masks - start_time_processing_masks:.4f} seconds")

            start_time_fusion = time.time()
            # Retrieve the individual point clouds and the fused point cloud of the objects
            pcs1, pcs2, fused_pc_objects = fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.3)
            end_time_fusion = time.time()
            timings["Point Cloud Fusion"].append(end_time_fusion - start_time_fusion)


            # Subtract the fused point cloud of the objects from the workspace point cloud
            fused_pc_objects_points = [pc for pc, _ in fused_pc_objects]

            # Combine the point clouds into a single numpy array
            if fused_pc_objects_points:
                fused_pc_objects_concatenated = np.vstack(fused_pc_objects_points)
            else:
                fused_pc_objects_concatenated = np.empty((0, 3))  # or handle the empty case appropriately

            print(f"Fused Point Cloud Objects Concatenated shape: {fused_pc_objects_concatenated.shape}")

            start_time_subtraction = time.time()
            # Subtract the fused point cloud of the objects from the workspace point cloud
            workspace_pc_subtracted = subtract_point_clouds_gpu(fused_point_cloud_ws, fused_pc_objects_concatenated, distance_threshold=0.06)
            # visualize_point_cloud(fused_point_cloud_ws, title="Fused Point Cloud Workspace")
            # visualize_point_cloud(workspace_pc_subtracted, title="Subtracted Point Cloud Workspace")
            end_time_subtraction = time.time()
            timings["Subtraction"].append(end_time_subtraction - start_time_subtraction)

            total_time = time.time() - start_time
            timings["Total Time per Iteration"].append(total_time)

            # Increment the frame count and calculate the FPS
            frame_count += 1
            # Record total loop time
            print("Total time per loop iteration: ", time.time() - start_time)
            fps = 1.0 / total_time
            fps_values.append(fps)
            current_timestamp = time.time()

            if len(fps_values) > 10:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            # Append the FPS and timestamp to the CSV file
            with open(fps_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_timestamp, fps])

            # Save the dictionary to a CSV file
            with open("timings.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Step", "Timings"])
                for step, values in timings.items():
                    writer.writerow([step, ",".join(map(str, values))])  # Save timings as comma-separated values

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
            combined_frame = cv2.resize(combined_frame, (combined_frame.shape[1] // 2, combined_frame.shape[0] // 2))
            cv2.imshow("YOLO11 Segmentation+Tracking", combined_frame)

            # Clear the point clouds from both cameras -> Prevent overflow
            point_clouds_camera1.clear()
            point_clouds_camera2.clear()
            pcs1.clear()
            pcs2.clear()
            fused_pc_objects.clear()
            # Wait for the 'q' key press to exit the loop
            key = cv2.waitKey(1)

    zed1.close()
    zed2.close()


if __name__ == "__main__":
    main()