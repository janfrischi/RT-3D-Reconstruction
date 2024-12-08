import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
import random
import open3d as o3d
import open3d.core as o3c
import cProfile
import pstats

from jsonschema.exceptions import best_match
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
    kernel = np.ones((12, 12), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

# Convert 2D mask pixel coordinates to 3D points using depth values on GPU
def convert_mask_to_3d_points(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    # Convert the mask indices (y,x coords where the mask is 1) to a tensor and move to GPU
    mask_indices = torch.tensor(mask_indices, device=device)
    # Extract the u, v coordinates from the mask indices
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]
    # Extract the depth values from the depth map and move to GPU
    depth_values = depth_map[v_coords, u_coords].to(device)
    # Create a mask to filter out invalid depth values
    valid_mask = (depth_values > 0) & ~torch.isnan(depth_values)
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

# TODO: This approach was not suitable for the current implementation as the objects don't overlap enough for the IoU metric to be useful
# Downsample a point cloud using voxel downsampling
# def downsample_point_cloud(point_cloud, voxel_size=0.005):
#     # Convert the numpy array to an Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud)
#     # Perform voxel downsampling
#     downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#     # Convert back to numpy array
#     downsampled_points = np.asarray(downsampled_pcd.points)
#     return downsampled_points

# Using this implementation we gain an additional 5fps
def downsample_point_cloud(point_cloud, voxel_size=0.005):
    # Convert the numpy array to an Open3D point cloud
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(point_cloud, device=o3c.Device("CUDA:0")))
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.005)
    downsampled_points = downsampled_pcd.point.positions.cpu().numpy()
    return downsampled_points


def filter_outliers_sor(point_cloud, nb_neighbors=20, std_ratio=5):
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
            print(f"Directly fused single object with class_id {class_id}")
            visualize_point_cloud(fused_pc, title=f"Fused Point Cloud for Class {class_id}")

        # If there are multiple point clouds with the same class ID from each camera, we need to find the best match
        else:
            for pc1 in pcs1:
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
                    fused_pc = filter_outliers_sor(np.vstack((pc1, best_match)))
                    fused_point_clouds.append((fused_pc, class_id))
                    # Remove the matched point cloud from the list of point clouds from camera 2 to prevent duplicate fusion
                    pcs2 = [pc for pc in pcs2 if not point_clouds_equal(pc, best_match)]
                    print(f"Fused based on centroid distance {best_distance} for class_id {class_id}")
                    visualize_point_cloud(fused_pc, title=f"Fused Point Cloud for Class {class_id}")

                # If no match was found, simply add the point cloud from camera 1 to the fused point clouds
                else:
                    fused_point_clouds.append((pc1, class_id))
                    print(f"No match found for Class {class_id}. Added original pc1.")
                    visualize_point_cloud(pc1, title=f"Original Point Cloud for Class {class_id}")

            # If any point clouds remain in the list from camera 2, add them to the fused point clouds
            for pc2 in pcs2:
                fused_point_clouds.append((pc2, class_id))
                print(f"Remaining pc2 added for Class {class_id}")

    return pcs1, pcs2, fused_point_clouds


def main():
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pre-trained YOLOv11 model and move it to the device
    model = YOLO("yolo11m-seg.pt").to(device)

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
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL # TODO: Check what if NEURAL and NEURAL_PLUS differ in performance
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
    # These matrices can be obtained from the extrinsic calibration process

    T_chess_cam1 = np.array([[0.8755, 0.2677, -0.4023, 0.5703],
                             [-0.4832, 0.4857, -0.7284, 1.1320],
                             [0.0004, 0.8321, 0.5546, -0.7219],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

    T_chess_cam2 = np.array([[0.5045, -0.4021, 0.7641, -0.8758],
                             [0.8633, 0.2236, -0.4524, 0.8031],
                             [0.0110, 0.8879, 0.4600, -0.7207],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

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

    distance_cam1 = np.linalg.norm(origin_cam1 - np.array([0, 0, 0]))
    distance_cam2 = np.linalg.norm(origin_cam2 - np.array([0, 0, 0]))

    print(f"Distance from robot frame to camera frame 1: {distance_cam1:.4f} meters")
    print(f"Distance from robot frame to camera frame 2: {distance_cam2:.4f} meters")

    # Initialize the image and depth map variables for both cameras
    image1 = sl.Mat()
    depth1 = sl.Mat()
    image2 = sl.Mat()
    depth2 = sl.Mat()

    # Initialize the key variable to check for the 'q' key press
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
                vid_stride = 15,
                classes=[39, 41],
                persist=True,
                retina_masks=True,
                conf=0.4,
                device=device,
                tracker="ultralytics/cfg/trackers/bytetrack.yaml"
            )

            results2 = model.track(
                source=frame2,
                imgsz=640,
                vid_stride=15,
                classes=[39, 41],
                persist=True,
                retina_masks=True,
                conf=0.4,
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

            # Processing the masks from camera 1
            if masks1 is not None:
                # Get depth-maps from both cameras
                depth_map1 = torch.from_numpy(zed_depth_np1).to(device)
                # Iterate over the masks and class IDs to extract the 3D points for each detected object
                for i, mask in enumerate(masks1.data):
                    mask = mask.cpu().numpy()
                    mask = erode_mask(mask, iterations=1)
                    # Move the mask indices to the GPU
                    mask_indices = np.argwhere(mask > 0)

                    # Calculate the 3D points using the mask indices and depth map -> This operation is done on the GPU
                    with torch.amp.autocast('cuda'):
                        points_3d = convert_mask_to_3d_points(mask_indices, depth_map1, cx1, cy1, fx1, fy1)

                    if points_3d.size(0) > 0:
                        # Move the 3D points to the CPU and transform them to the robot base frame
                        point_cloud_cam1 = points_3d.cpu().numpy()
                        point_cloud_cam1_transformed = np.dot(rotation_robot_cam1, point_cloud_cam1.T).T + origin_cam1
                        # Downsampling the point cloud
                        point_cloud_cam1_transformed = downsample_point_cloud(point_cloud_cam1_transformed, voxel_size=0.01)
                        # Add the downsampled point cloud and class ID to this cameras point cloud list
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

                    with torch.cuda.amp.autocast():
                        points_3d = convert_mask_to_3d_points(mask_indices, depth_map2, cx2, cy2, fx2, fy2)

                    if points_3d.size(0) > 0:
                        point_cloud_cam2 = points_3d.cpu().numpy()
                        point_cloud_cam2_transformed = np.dot(rotation_robot_cam2, point_cloud_cam2.T).T + origin_cam2

                        # Down sampling the point cloud
                        point_cloud_cam2_transformed = downsample_point_cloud(point_cloud_cam2_transformed, voxel_size=0.01)
                        point_clouds_camera2.append((point_cloud_cam2_transformed, int(class_ids2[i])))
                        print(f"Class ID: {class_ids2[i]} ({class_names[class_ids2[i]]}) in Camera Frame 2")

            # Retrieve the individual point clouds and the fused point cloud
            pcs1, pcs2, fused_pc = fuse_point_clouds_centroid(point_clouds_camera1, point_clouds_camera2, distance_threshold=0.3)

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
            combined_frame = cv2.resize(combined_frame, (combined_frame.shape[1] // 2, combined_frame.shape[0] // 2))
            cv2.imshow("YOLO11 Segmentation+Tracking", combined_frame)

            # Clear the point clouds from both cameras
            point_clouds_camera1.clear()
            point_clouds_camera2.clear()
            pcs1.clear()
            pcs2.clear()
            fused_pc.clear()

            key = cv2.waitKey(1)

    zed1.close()
    zed2.close()


if __name__ == "__main__":
    main()