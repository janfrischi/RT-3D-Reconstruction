import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import torch.nn.functional as F
import open3d as o3d
import csv
from ultralytics import YOLO


def erode_mask_gpu(mask, kernel_size=3):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    eroded_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=kernel_size // 2)
    return (eroded_mask.squeeze() > 0).float()


# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def convert_mask_to_3d_points(mask_indices, depth_map, cx, cy, fx, fy):
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


# Perform down sampling directly on the GPU
def downsample_point_cloud_gpu(point_cloud, voxel_size):
    # Round the point cloud to the nearest voxel grid -> all points in the same voxel will have the same coordinates
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


def calculate_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)


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
            fused_pc = filter_outliers_sor(np.vstack((pcs1[0], pcs2[0])))
            fused_point_clouds.append((fused_pc, class_id))

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
                        best_match = filter_outliers_sor(best_match)

                # If a match was found, fuse the point clouds
                if best_match is not None:
                    # Concatenate the point clouds along the vertical axis and filter out the outliers
                    fused_pc = np.vstack((pc1, best_match))
                    fused_point_clouds.append((fused_pc, class_id))
                    # Remove the matched point cloud from the list of point clouds from camera 2 to prevent duplicate fusion
                    pcs2 = [pc for pc in pcs2 if not point_clouds_equal(pc, best_match)]

                # If no match was found, simply add the point cloud from camera 1 to the fused point clouds
                else:
                    fused_point_clouds.append((pc1, class_id))

            # If any point clouds remain in the list from camera 2, add them to the fused point clouds
            for pc2 in pcs2:
                fused_point_clouds.append((pc2, class_id))

    return pcs1, pcs2, fused_point_clouds


def subtract_point_clouds_gpu(workspace_pc, objects_pc, distance_threshold=0.005):
    # Convert point clouds to PyTorch tensors and ensure consistent dtype
    workspace_tensor = torch.tensor(workspace_pc, dtype=torch.float32, device='cuda')
    objects_tensor = torch.tensor(objects_pc, dtype=torch.float32, device='cuda')

    # Compute pairwise distances, torch.cdist computes the pairwise Euclidean distances between two tensors
    # distances[i,j] contains the ith point in workspace_tensor and the jth point in objects_tensor
    distances = torch.cdist(workspace_tensor, objects_tensor)

    print(f"Memory allocated for distances: {distances.element_size() * distances.nelement() / 1024 / 1024:.2f} MB")

    # Find points in the workspace tensor that are farther than the threshold from all points in the objects tensor
    min_distances, _ = distances.min(dim=1)
    # If min_distances[i] > distance_threshold, the ith point is retained
    mask = min_distances > distance_threshold

    # Filter the workspace points based on the mask
    filtered_points = workspace_tensor[mask].cpu().numpy()

    return filtered_points


def retrieve_frames(zed1, zed2, image1, image2, timings):
    # Record the start time for frame retrieval
    retrieval_start_time = time.time()

    # Retrieve images from both ZED cameras
    zed1.retrieve_image(image1, view=sl.VIEW.LEFT)
    zed2.retrieve_image(image2, view=sl.VIEW.LEFT)

    # Get the image data from the retrieved images
    frame1 = image1.get_data()
    frame2 = image2.get_data()

    # Convert the images from BGRA to BGR color space
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)

    # Record the end time for frame retrieval
    retrieval_end_time = time.time()

    # Calculate and store the time taken for frame retrieval
    timings["Frame Retrieval"].append(retrieval_end_time - retrieval_start_time)

    # Print the time taken for frame retrieval
    print(f"Frame retrieval time: {retrieval_end_time - retrieval_start_time:.4f} seconds")

    # Return the retrieved frames
    return frame1, frame2

def retrieve_depth_maps(zed1, zed2, depth1, depth2, timings):
    depth_start_time = time.time()
    depth_retrieval_result1 = zed1.retrieve_measure(depth1, measure=sl.MEASURE.DEPTH)
    depth_retrieval_result2 = zed2.retrieve_measure(depth2, measure=sl.MEASURE.DEPTH)
    zed_depth_np1 = depth1.get_data()
    zed_depth_np2 = depth2.get_data()
    depth_end_time = time.time()
    timings["Depth Retrieval"].append(depth_end_time - depth_start_time)
    print(f"Depth retrieval time: {depth_end_time - depth_start_time:.4f} seconds")
    return depth_retrieval_result1, depth_retrieval_result2, zed_depth_np1, zed_depth_np2

def process_point_clouds(zed1, zed2, point_cloud1_ws, point_cloud2_ws, resolution, rotation1_torch, origin1_torch, rotation2_torch, origin2_torch, device, timings):
    point_cloud_start_time = time.time()
    zed1.retrieve_measure(point_cloud1_ws, measure=sl.MEASURE.XYZ, resolution=resolution)
    zed2.retrieve_measure(point_cloud2_ws, measure=sl.MEASURE.XYZ, resolution=resolution)
    point_cloud1_ws_tensor = torch.tensor(point_cloud1_ws.get_data()[:, :, :3], dtype=torch.float32, device=device).reshape(-1, 3)
    point_cloud2_ws_tensor = torch.tensor(point_cloud2_ws.get_data()[:, :, :3], dtype=torch.float32, device=device).reshape(-1, 3)
    valid_mask_workspace1 = torch.isfinite(point_cloud1_ws_tensor).all(dim=1)
    point_cloud1_ws_tensor = point_cloud1_ws_tensor[valid_mask_workspace1]
    valid_mask_workspace2 = torch.isfinite(point_cloud2_ws_tensor).all(dim=1)
    point_cloud2_ws_tensor = point_cloud2_ws_tensor[valid_mask_workspace2]
    point_cloud1_ws_transformed_tensor = torch.mm(rotation1_torch, point_cloud1_ws_tensor.T).T + origin1_torch
    point_cloud2_ws_transformed_tensor = torch.mm(rotation2_torch, point_cloud2_ws_tensor.T).T + origin2_torch
    x_bounds_baseframe = (-0.25, 0.75)
    y_bounds_baseframe = (-0.5, 1.75)
    z_bounds_baseframe = (-0.05, 2)
    point_cloud1_workspace_np_cropped = crop_point_cloud_gpu(point_cloud1_ws_transformed_tensor, x_bounds_baseframe, y_bounds_baseframe, z_bounds_baseframe)
    point_cloud2_workspace_np_cropped = crop_point_cloud_gpu(point_cloud2_ws_transformed_tensor, x_bounds_baseframe, y_bounds_baseframe, z_bounds_baseframe)
    point_cloud1_workspace = downsample_point_cloud_gpu(point_cloud1_workspace_np_cropped, voxel_size=0.005)
    point_cloud2_workspace = downsample_point_cloud_gpu(point_cloud2_workspace_np_cropped, voxel_size=0.005)
    fused_point_cloud_ws = torch.cat((point_cloud1_workspace, point_cloud2_workspace), dim=0)
    fused_point_cloud_ws = fused_point_cloud_ws.cpu().numpy()
    print(f"Fused Point Cloud Workspace shape: {fused_point_cloud_ws.shape}")
    point_cloud_end_time = time.time()
    timings["Point Cloud Processing"].append(point_cloud_end_time - point_cloud_start_time)
    print(f"Point cloud processing time: {point_cloud_end_time - point_cloud_start_time:.4f} seconds")
    return fused_point_cloud_ws

def perform_yolo_inference(model, frame1, frame2, device, timings):
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

    annotated_frame1 = yolo11_results1[0].plot(line_width=2, font_size=18)
    annotated_frame2 = yolo11_results2[0].plot(line_width=2, font_size=18)

    masks1 = yolo11_results1[0].masks
    masks2 = yolo11_results2[0].masks
    class_ids1 = yolo11_results1[0].boxes.cls.cpu().numpy()
    class_ids2 = yolo11_results2[0].boxes.cls.cpu().numpy()
    yolo11_end_time = time.time()
    timings["YOLO11 Inference"].append(yolo11_end_time - yolo11_start_time)
    print(f"YOLO11 processing time: {yolo11_end_time - yolo11_start_time:.4f} seconds")

    return annotated_frame1, annotated_frame2, masks1, masks2, class_ids1, class_ids2

def process_masks(masks, class_ids, zed_depth_np, cx, cy, fx, fy, rotation_robot_cam_torch, origin_cam_torch, device, point_clouds_camera):
    if masks is not None:
        depth_map = torch.tensor(zed_depth_np, dtype=torch.float32, device=device)
        for i, mask in enumerate(masks.data):
            mask_indices = torch.nonzero(mask, as_tuple=False)
            with torch.amp.autocast('cuda'):
                points_3d_cam = convert_mask_to_3d_points(mask_indices, depth_map, cx, cy, fx, fy)

            if points_3d_cam.size(0) > 0:
                point_cloud_cam_transformed = torch.mm(points_3d_cam, rotation_robot_cam_torch.T) + origin_cam_torch
                point_cloud_cam_downsampled = downsample_point_cloud_gpu(point_cloud_cam_transformed, voxel_size=0.005)
                point_cloud_cam_downsampled_cpu = point_cloud_cam_downsampled.cpu().numpy()
                point_clouds_camera.append((point_cloud_cam_downsampled_cpu, int(class_ids[i])))


def fuse_point_clouds(point_clouds_camera1, point_clouds_camera2, distance_threshold, timings):
    start_time_fusion = time.time()
    pcs1, pcs2, fused_pc_objects = fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold)
    end_time_fusion = time.time()
    timings["Point Cloud Fusion"].append(end_time_fusion - start_time_fusion)

    fused_pc_objects_points = [pc for pc, _ in fused_pc_objects]
    if fused_pc_objects_points:
        fused_pc_objects_concatenated = np.vstack(fused_pc_objects_points)
    else:
        fused_pc_objects_concatenated = np.empty((0, 3))

    print(f"Fused Point Cloud Objects Concatenated shape: {fused_pc_objects_concatenated.shape}")
    return pcs1, pcs2, fused_pc_objects_concatenated

def subtract_point_clouds(fused_point_cloud_ws, fused_pc_objects_concatenated, distance_threshold, timings):
    start_time_subtraction = time.time()
    workspace_pc_subtracted = subtract_point_clouds_gpu(fused_point_cloud_ws, fused_pc_objects_concatenated, distance_threshold)
    end_time_subtraction = time.time()
    timings["Subtraction"].append(end_time_subtraction - start_time_subtraction)
    return workspace_pc_subtracted

def process_iteration_end(start_time, timings, frame_count, fps_values, fps_log_file, annotated_frame1, annotated_frame2, point_clouds_camera1, point_clouds_camera2, pcs1, pcs2):
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
    with open("../timings.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Step", "Timings"])
        for step, values in timings.items():
            writer.writerow([step, ",".join(map(str, values))])  # Save timings as comma-separated values

    # Display the annotated frames with the FPS
    if annotated_frame1 is not None:
        cv2.putText(annotated_frame1, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        height, width, _ = annotated_frame1.shape
        cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
        cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame1)

    if annotated_frame2 is not None:
        cv2.putText(annotated_frame2, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        height, width, _ = annotated_frame2.shape
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

    return frame_count, fps_values

