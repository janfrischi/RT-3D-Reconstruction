import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import torch.nn.functional as F
import open3d as o3d
import open3d.core as o3c
import matplotlib.pyplot as plt
import csv
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

# Dictionary to store cumulative timings
timings = {
    "Frame Retrieval": [],
    "Depth Retrieval": [],
    "Point Cloud Processing": [],
    "YOLO11 Inference": [],
    "Mask Processing": [],
    "Point Cloud Fusion": [],
    "Point Cloud Subtraction": [],
    "Total Time per Iteration": []
}

# TODO: Check if this function is needed
def erode_mask_gpu(mask, kernel_size=12):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    eroded_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=kernel_size // 2)
    return (eroded_mask.squeeze() > 0).float()


# Erode the mask to remove background noise, since masks aren't perfect
def erode_mask(mask, iterations=1):
    kernel = np.ones((12, 12), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

# Using this implementation we gain an additional 5fps
def downsample_point_cloud(point_cloud, voxel_size=0.01):
    # Convert the numpy array to an Open3D point cloud
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(point_cloud, device=o3c.Device("CUDA:0")))
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = downsampled_pcd.point.positions.cpu().numpy()
    return downsampled_points


# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def convert_mask_to_3d_points_gpu(mask_indices, depth_map, cx, cy, fx, fy):
    # Extract the u, v coordinates from the mask indices
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    # Extract the necessary depth values from the depth map
    depth_values = depth_map[v_coords, u_coords]
    # Create a mask to filter out invalid depth values
    valid_mask = (depth_values > 0) & ~torch.isnan(depth_values) & ~torch.isinf(depth_values)
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

# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def convert_mask_to_3d_points_cpu(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    # Convert the mask indices (y,x coords where the mask is 1) to a tensor and move to GPU
    mask_indices = torch.tensor(mask_indices, device=device)
    # Extract the u, v coordinates from the mask indices
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    # Extract the necessary depth values from the depth map and move to GPU
    depth_values = depth_map[v_coords, u_coords].to(device)
    # Create a mask to filter out invalid depth values
    valid_mask = (depth_values > 0) & ~torch.isnan(depth_values) & ~torch.isinf(depth_values)
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


# Perform down sampling directly on the GPU
def downsample_point_cloud_gpu(point_cloud, voxel_size):
    rounded = torch.round(point_cloud / voxel_size) * voxel_size
    downsampled_points = torch.unique(rounded, dim=0)
    return downsampled_points


# Filter out the outliers in the point cloud using the Statistical Outlier Removal filter
def filter_outliers_sor(point_cloud, nb_neighbors=20, std_ratio=1.5):
    # Convert the numpy array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Apply statistical outlier removal
    filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Convert back to numpy array
    filtered_points = np.asarray(filtered_pcd.points)
    return filtered_points


# Returns true if the two point clouds are equal
def point_clouds_equal(pc1, pc2):
    return np.array_equal(pc1, pc2)


# Helper function to visualize a point cloud
def visualize_point_cloud(point_cloud, title="Point Cloud"):
    # Create an Open3D PointCloud object and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd], window_name=title)

# CPU implementation of the function to calculate the centroid of a point cloud
def calculate_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)

# GPU implementation of the function to calculate the centroid of a point cloud
def calculate_centroid_gpu(point_cloud):
    return torch.mean(point_cloud, dim=0)

# GPU implementation of the function to calculate the centroid of a point cloud
def crop_point_cloud_gpu(point_cloud, x_bounds, y_bounds, z_bounds):
    mask = (
        (point_cloud[:, 0] >= x_bounds[0]) & (point_cloud[:, 0] <= x_bounds[1]) &
        (point_cloud[:, 1] >= y_bounds[0]) & (point_cloud[:, 1] <= y_bounds[1]) &
        (point_cloud[:, 2] >= z_bounds[0]) & (point_cloud[:, 2] <= z_bounds[1])
    )
    return point_cloud[mask]


# Function to fuse the point clouds based on centroid distance
def fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.1):
    # Group the point clouds by the class ID
    # The class dicts are of the form: {class_id1: [point_cloud1, point_cloud2, ...], class_id2: [point_cloud1, point_cloud2, ...]}
    pcs1 = []
    pcs2 = []
    class_dict1 = {}
    class_dict2 = {}

    # Iterate over the point clouds from camera 1 and group them by class ID
    # point_clouds_camera1 = [(pc, class_id), ...] is a list of tuples containing the point cloud and the class ID
    for pc, class_id in point_clouds_camera1:
        if class_id not in class_dict1:
            class_dict1[class_id] = [] # To store the point clouds for the class ID
        class_dict1[class_id].append(pc) # Append the point cloud to the list for the class ID

    # Iterate over the point clouds from camera 2 and group them by class ID
    for pc, class_id in point_clouds_camera2:
        if class_id not in class_dict2:
            class_dict2[class_id] = []
        class_dict2[class_id].append(pc)

    # After this loop, class_dict1 and class_dict2 contain the point clouds grouped by class ID

    # Initialize the fused point cloud list
    fused_point_clouds = []

    # Process each class ID
    # Get all the unique class IDs from both cameras
    # class_dict1.keys() returns a set of all the keys "class IDs" in the dictionary
    for class_id in set(class_dict1.keys()).union(class_dict2.keys()):
        # Get the point clouds for the current class ID from both cameras
        pcs1 = class_dict1.get(class_id, []) # pcs1 has the following format: [point_cloud1, point_cloud2, ...]
        pcs2 = class_dict2.get(class_id, [])

        # If there is only one point cloud with the same class ID from each camera we can directly fuse the pcs
        if len(pcs1) == 1 and len(pcs2) == 1:
            # Concatenate the point clouds along the vertical axis
            fused_pc = np.vstack((pcs1[0], pcs2[0]))
            fused_point_clouds.append((fused_pc, class_id))
            print(f"Directly fused single object with class_id {class_id}")

        # If there are multiple point clouds with the same class ID from each camera, we need to find the best match
        else:
            for pc1 in pcs1:
                pc1 = filter_outliers_sor(pc1)
                best_distance = float('inf')
                best_match = None

                # Calculate the centroid of the point cloud from camera 1
                centroid1 = calculate_centroid(pc1)

                # Loop over all the point clouds from camera 2 with the same ID and find the best match based on centroid distance
                for pc2 in pcs2:
                    centroid2 = calculate_centroid(pc2)
                    # Calculate the Euclidean distance / L2 norm between the centroids
                    distance = np.linalg.norm(centroid1 - centroid2)

                    if distance < best_distance and distance < distance_threshold:
                        best_distance = distance
                        best_match = pc2

                # If a match was found, fuse the point clouds
                if best_match is not None:
                    # Concatenate the point clouds along the vertical axis and filter out the outliers
                    fused_pc = np.vstack((pc1, best_match))
                    fused_point_clouds.append((fused_pc, class_id))
                    # Remove the matched point cloud from the list of point clouds from camera 2 to prevent duplicate fusion
                    pcs2 = [pc for pc in pcs2 if not point_clouds_equal(pc, best_match)]
                    print(f"Fused based on centroid distance {best_distance} for class_id {class_id}")

                # If no match was found, simply add the point cloud from camera 1 to the fused point clouds
                else:
                    fused_point_clouds.append((pc1, class_id))
                    print(f"No match found for Class {class_id}. Added original pc1.")

            # If any point clouds remain in the list from camera 2, add them to the fused point clouds
            for pc2 in pcs2:
                fused_point_clouds.append((pc2, class_id))
                print(f"Remaining pc2 added for Class {class_id}")

    return pcs1, pcs2, fused_point_clouds


def subtract_point_clouds(workspace_pc, objects_pc, distance_threshold=0.01):
    # Convert the numpy arrays to Open3D point cloud objects
    workspace_pcd = o3d.geometry.PointCloud()
    workspace_pcd.points = o3d.utility.Vector3dVector(workspace_pc)

    objects_pcd = o3d.geometry.PointCloud()
    objects_pcd.points = o3d.utility.Vector3dVector(objects_pc)

    # Create a KDTree for the objects point cloud
    kdtree = o3d.geometry.KDTreeFlann(objects_pcd)

    # Find and remove points in the workspace point cloud that are close to points in the objects point cloud
    indices_to_remove = []
    for i, point in enumerate(workspace_pcd.points):
        [_, idx, _] = kdtree.search_radius_vector_3d(point, distance_threshold)
        if len(idx) > 0:
            indices_to_remove.append(i)

    # Remove the points
    workspace_pcd = workspace_pcd.select_by_index(indices_to_remove, invert=True)

    # Convert back to numpy array
    workspace_pc_subtracted = np.asarray(workspace_pcd.points)

    return workspace_pc_subtracted

def subtract_point_clouds_gpu(workspace_pc, objects_pc, distance_threshold=0.005):
    # Convert point clouds to PyTorch tensors and ensure consistent dtype
    workspace_tensor = torch.tensor(workspace_pc, dtype=torch.float32, device='cuda')
    objects_tensor = torch.tensor(objects_pc, dtype=torch.float32, device='cuda')

    # Compute pairwise distances
    distances = torch.cdist(workspace_tensor, objects_tensor)

    # Find points in the workspace tensor that are farther than the threshold from all points in the objects tensor
    min_distances, _ = distances.min(dim=1)
    mask = min_distances > distance_threshold

    # Filter the workspace points based on the mask
    filtered_points = workspace_tensor[mask].cpu().numpy()

    return filtered_points


def voxel_grid_subtract(workspace_pc, objects_pc, voxel_size=0.02):
    # Convert NumPy arrays to Open3D PointCloud objects
    workspace_o3d = o3d.geometry.PointCloud()
    workspace_o3d.points = o3d.utility.Vector3dVector(workspace_pc)

    objects_o3d = o3d.geometry.PointCloud()
    objects_o3d.points = o3d.utility.Vector3dVector(objects_pc)

    # Apply voxelization to both point clouds
    workspace_voxelized = workspace_o3d.voxel_down_sample(voxel_size=voxel_size)
    objects_voxelized = objects_o3d.voxel_down_sample(voxel_size=voxel_size)

    # Get voxel grids as sets of unique voxel indices
    workspace_voxels = set(map(tuple, np.asarray(workspace_voxelized.points)))
    objects_voxels = set(map(tuple, np.asarray(objects_voxelized.points)))

    # Subtract the object voxels from the workspace voxels
    filtered_voxels = np.array(list(workspace_voxels - objects_voxels))

    return filtered_voxels


def main():
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the CSV file to store the results
    fps_log_file = "fps_log.csv"

    # Create or overwrite the CSV file and write the header
    with open(fps_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "FPS"])

    # Load the pre-trained YOLOv11 model and move it to the device
    model = YOLO("yolo11l-seg.pt").to(device)

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
    point_cloud_resolution = sl.Resolution(640, 360)

    # Initialize the image and depth map objects for both cameras
    image1 = sl.Mat()
    depth1 = sl.Mat()
    image2 = sl.Mat()
    depth2 = sl.Mat()

    # Create point cloud objects to hold data of the workspace
    point_cloud1_ws = sl.Mat(point_cloud_resolution.width, point_cloud_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud2_ws = sl.Mat(point_cloud_resolution.width, point_cloud_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

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

            # Check if the depth maps were successfully retrieved
            if depth_retrieval_result1 != sl.ERROR_CODE.SUCCESS or depth_retrieval_result2 != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result1}, {depth_retrieval_result2}")
                continue

            depth_end_time = time.time()
            timings["Depth Retrieval"].append(depth_end_time - depth_start_time)
            print(f"Depth retrieval time: {depth_end_time - depth_start_time:.4f} seconds")

            # Step 3: Point cloud retrieval and processing
            point_cloud_start_time = time.time()

            # 1. Retrieve the point clouds for the workspace from both cameras
            zed1.retrieve_measure(point_cloud1_ws, measure=sl.MEASURE.XYZ, resolution=point_cloud_resolution)
            zed2.retrieve_measure(point_cloud2_ws, measure=sl.MEASURE.XYZ, resolution=point_cloud_resolution)

            # 2. Convert the point clouds to tensors directly
            point_cloud1_ws_tensor = torch.tensor(point_cloud1_ws.get_data()[:, :, :3], dtype=torch.float32,
                                                  device=device).reshape(-1, 3)
            point_cloud2_ws_tensor = torch.tensor(point_cloud2_ws.get_data()[:, :, :3], dtype=torch.float32,
                                                  device=device).reshape(-1, 3)

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

            # Crop the point clouds to the workspace
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
            # TODO: Check how the voxel size affects the performance of the pipeline
            point_cloud1_workspace = downsample_point_cloud_gpu(point_cloud1_workspace_np_cropped, voxel_size=0.005)
            point_cloud2_workspace = downsample_point_cloud_gpu(point_cloud2_workspace_np_cropped, voxel_size=0.005)
            fused_point_cloud_ws = torch.cat((point_cloud1_workspace, point_cloud2_workspace), dim=0)
            # Convert to numpy array
            fused_point_cloud_ws = fused_point_cloud_ws.cpu().numpy()
            visualize_point_cloud(fused_point_cloud_ws, title="Fused Workspace before SOR")
            # SOR removal
            fused_point_cloud_ws = filter_outliers_sor(fused_point_cloud_ws, nb_neighbors=20, std_ratio=1.5)
            visualize_point_cloud(fused_point_cloud_ws, title="Fused Workspace after SOR")

            point_cloud_end_time = time.time()
            timings["Point Cloud Processing"].append(point_cloud_end_time - point_cloud_start_time)
            print(f"Point cloud processing time: {point_cloud_end_time - point_cloud_start_time:.4f} seconds")

            # Step 4: Object detection and segmentation using YOLO11
            yolo11_start_time = time.time()

            yolo11_results1 = model.track(
                source=frame1,
                imgsz=640,
                classes=[39, 41, 64, 66, 73],
                persist=True,
                retina_masks=True,
                conf=0.25,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            yolo11_results2 = model.track(
                source=frame2,
                imgsz=640,
                classes=[39, 41, 64, 66, 73],
                persist=True,
                retina_masks=True,
                conf=0.25,
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

            # Step 5: Mask processing
            start_time_processing_masks = time.time()
            # Processing the masks from camera 1
            if masks1 is not None:
                # Get depth-maps from both cameras, convert input Numpy arrays to PyTorch tensors and move them to the GPU
                depth_map1 = torch.from_numpy(zed_depth_np1).to(device)
                # Iterate over the masks and class IDs to extract the 3D points for each detected object
                for i, mask in enumerate(masks1.data):
                    mask = mask.cpu().numpy()
                    mask = erode_mask(mask, iterations=1)
                    # Get the indices of the mask where the mask is 1
                    mask_indices = np.argwhere(mask > 0)

                    # Calculate the 3D points using the mask indices and depth map -> This operation is done on the GPU
                    with torch.amp.autocast('cuda'):
                        points_3d_cam1 = convert_mask_to_3d_points_cpu(mask_indices, depth_map1, cx1, cy1, fx1, fy1)

                    if points_3d_cam1.size(0) > 0:
                        # Move the 3D points to the CPU and transform them to the robot base frame, NumPy operations are done on the CPU
                        point_cloud_cam1 = points_3d_cam1.cpu().numpy()
                        point_cloud_cam1_transformed = np.dot(rotation_robot_cam1, point_cloud_cam1.T).T + origin_cam1
                        # Down sampling the point cloud
                        point_cloud_cam1_transformed = downsample_point_cloud(point_cloud_cam1_transformed,
                                                                              voxel_size=0.005)
                        # Add the down sampled point cloud and class ID to this cameras point cloud list
                        point_clouds_camera1.append((point_cloud_cam1_transformed, int(class_ids1[i])))
                        print(f"Class ID: {class_ids1[i]} ({class_names[class_ids1[i]]}) in Camera Frame 1")

            # Processing the masks from camera 2
            if masks2 is not None:
                # Get depth-maps from both cameras
                depth_map2 = torch.from_numpy(zed_depth_np2).to(device)
                # Iterate over the masks and class IDs to extract the 3D points for each detected object
                for i, mask in enumerate(masks2.data):
                    mask = mask.cpu().numpy()
                    mask = erode_mask(mask, iterations=1)
                    mask_indices = np.argwhere(mask > 0)

                    with torch.amp.autocast('cuda'):
                        points_3d_cam2 = convert_mask_to_3d_points_cpu(mask_indices, depth_map2, cx2, cy2, fx2, fy2)

                    if points_3d_cam2.size(0) > 0:
                        point_cloud_cam2 = points_3d_cam2.cpu().numpy()
                        point_cloud_cam2_transformed = np.dot(rotation_robot_cam2, point_cloud_cam2.T).T + origin_cam2

                        # Down sampling the point cloud
                        point_cloud_cam2_transformed = downsample_point_cloud(point_cloud_cam2_transformed,
                                                                              voxel_size=0.005)
                        point_clouds_camera2.append((point_cloud_cam2_transformed, int(class_ids2[i])))
                        print(f"Class ID: {class_ids2[i]} ({class_names[class_ids2[i]]}) in Camera Frame 2")



            end_time_processing_masks = time.time()
            timings["Mask Processing"].append(end_time_processing_masks - start_time_processing_masks)
            print(f"Processing masks time: {end_time_processing_masks - start_time_processing_masks:.4f} seconds")

            # Step 6: Point cloud fusion and subtraction
            start_time_fusion = time.time()
            # Retrieve the individual point clouds and the fused point cloud of the objects
            pcs1, pcs2, fused_pc_objects = fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.3)

            end_time_fusion = time.time()
            timings["Point Cloud Fusion"].append(end_time_fusion - start_time_fusion)

            # Step 7: Point cloud subtraction
            start_time_subtraction = time.time()

            # Extract the point clouds from the fused_pc_objects list
            fused_pc_objects_points = [pc for pc, _ in fused_pc_objects]

            # Combine the point clouds into a single numpy array
            if fused_pc_objects_points:
                fused_pc_objects_concatenated = np.vstack(fused_pc_objects_points)
            else:
                fused_pc_objects_concatenated = np.empty((0, 3))  # or handle the empty case appropriately

            # Subtract the fused point cloud of the objects from the workspace point cloud -> Using KDTree
            point_cloud_ws_subtracted = subtract_point_clouds(fused_point_cloud_ws, fused_pc_objects_concatenated, distance_threshold=0.06)
            print(f"Workspace Point Cloud Subtracted shape - KDTree: {point_cloud_ws_subtracted.shape}")

            point_cloud_ws_subtracted_1 = subtract_point_clouds_gpu(fused_point_cloud_ws, fused_pc_objects_concatenated, distance_threshold=0.06)

            # Visualize the subtracted point cloud
            visualize_point_cloud(point_cloud_ws_subtracted, title="Workspace Point Cloud Subtracted - KDTree")
            visualize_point_cloud(point_cloud_ws_subtracted_1, title="Workspace Point Cloud Subtracted - GPU")

            end_time_subtraction = time.time()
            timings["Point Cloud Subtraction"].append(end_time_subtraction - start_time_subtraction)

            total_time = time.time() - start_time - 0.003
            timings["Total Time per Iteration"].append(total_time)

            # Increment the frame count and calculate the FPS
            frame_count += 1
            fps = 1.0 / (time.time() - start_time)
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