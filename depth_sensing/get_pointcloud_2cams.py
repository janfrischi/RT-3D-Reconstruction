import sys
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import open3d.core as o3c

# This function is already part of the object_detection_and_point_cloud_processing class
def downsample_point_cloud(point_cloud, voxel_size=0.01):
    """
    Downsample a point cloud using voxel downsampling with Open3D.

    Args:
        point_cloud (np.ndarray): The input point cloud of shape (N, 3).
        voxel_size (float): The size of the voxel for downsampling.

    Returns:
        np.ndarray: The downsampled point cloud of shape (M, 3).
    """
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(point_cloud, device=o3c.Device("CUDA:0")))
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd.point.positions.cpu().numpy()


def retrieve_point_cloud_workspace(zed, point_cloud_mat, resolution):
    """
    Retrieve the point cloud from a ZED camera and convert it to a NumPy array.

    Args:
        zed (sl.Camera): The ZED camera object.
        point_cloud_mat (sl.Mat): The ZED Mat object for storing the point cloud.
        resolution (sl.Resolution): The resolution of the point cloud.

    Returns:
        np.ndarray: The retrieved point cloud as a NumPy array of shape (N, 3).
    """
    # Grab a frame from the ZED camera
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the point cloud with color
        zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)

        # Convert the point cloud to a NumPy array
        point_cloud_workspace_np = point_cloud_mat.get_data()[:, :, :3]  # Extract only XYZ, ignoring RGBA

        # Reshape to (N, 3), where N is the number of points
        point_cloud_workspace_np = point_cloud_workspace_np.reshape(-1, 3)

        # Remove invalid points (NaN or Inf values) -> Only keep valid points for further processing
        # The any(axis=1) method checks if any value in a row is NaN or Inf, ~ is the negation operator. True becomes False and vice versa
        valid_mask_workspace = ~np.isnan(point_cloud_workspace_np).any(axis=1) & ~np.isinf(point_cloud_workspace_np).any(axis=1)

        return point_cloud_workspace_np[valid_mask_workspace]

    return None


def main():
    print("Running Live Point Cloud Retrieval with Two Cameras... Press 'Ctrl+C' to quit")

    # Serial numbers of the two cameras
    sn_cam1 = 33137761
    sn_cam2 = 36829049

    # Initialize ZED cameras with their respective serial numbers
    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params1.coordinate_units = sl.UNIT.METER

    init_params2 = sl.InitParameters()
    init_params2.set_from_serial_number(sn_cam2)
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 30
    init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params2.depth_minimum_distance = 0.4
    init_params2.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params2.coordinate_units = sl.UNIT.METER

    zed1 = sl.Camera()
    zed2 = sl.Camera()

    # Open the cameras
    status1 = zed1.open(init_params1)
    if status1 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 1: {repr(status1)}")
        exit()

    status2 = zed2.open(init_params2)
    if status2 != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera 2: {repr(status2)}")
        exit()

    # Set resolution for the point clouds
    resolution = sl.Resolution(1280, 720)

    # Create point cloud objects to hold data
    point_cloud1 = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud2 = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    try:
        while True:
            # Retrieve point clouds from both cameras
            point_cloud_ws_cam1 = retrieve_point_cloud_workspace(zed1, point_cloud1, resolution)
            point_cloud_ws_cam2 = retrieve_point_cloud_workspace(zed2, point_cloud2, resolution)

            if point_cloud_ws_cam1 is not None:
                point_cloud_ws_cam1_downsampled = downsample_point_cloud(point_cloud_ws_cam1)
                print(f"Camera 1: Downsampled Point Cloud shape: {point_cloud_ws_cam1_downsampled.shape}")

            if point_cloud_ws_cam2 is not None:
                point_cloud_ws_cam1_downsampled = downsample_point_cloud(point_cloud_ws_cam2)
                print(f"Camera 2: Downsampled Point Cloud shape: {point_cloud_ws_cam1_downsampled.shape}")

    except KeyboardInterrupt:
        print("Exiting...")

    # Cleanup
    zed1.close()
    zed2.close()


if __name__ == "__main__":
    main()
