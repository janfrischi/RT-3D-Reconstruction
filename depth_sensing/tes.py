import sys
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import open3d.core as o3c

def downsample_point_cloud(point_cloud, voxel_size=0.01):
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(point_cloud, device=o3c.Device("CUDA:0")))
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd.point.positions.cpu().numpy()


def main():
    print("Running Live Point Cloud Retrieval... Press 'Ctrl+C' to quit")

    # Initialize the ZED camera
    sn_cam1 = 33137761
    sn_cam2 = 36829049
    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Convention for OpenGL
    init_params1.coordinate_units = sl.UNIT.METER
    zed = sl.Camera()

    status = zed.open(init_params1)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Set resolution for the point cloud
    resolution = sl.Resolution(1280, 720)  # Example resolution, adjust as needed

    # Create a point cloud object to hold data
    point_cloud = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    try:
        while True:
            # Grab frames from the ZED camera
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve the point cloud with color
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)

                # Convert the point cloud to a NumPy array
                 # Get the raw data
                point_cloud_np = point_cloud.get_data()[:, :, :3]  # Extract only XYZ, ignoring RGBA

                # Reshape to (N, 3), where N is the number of points
                point_cloud_np = point_cloud_np.reshape(-1, 3)

                # Remove invalid points (NaN or Inf values)
                valid_mask = ~np.isnan(point_cloud_np).any(axis=1) & ~np.isinf(point_cloud_np).any(axis=1)
                point_cloud_np = point_cloud_np[valid_mask]
                down_sampled_pointcloud = downsample_point_cloud(point_cloud_np)


                # Point cloud is now a clean NumPy array
                print(f"Downsampled Point cloud shape: {down_sampled_pointcloud.shape}")
                print(f"Point cloud shape: {point_cloud_np.shape}")

    except KeyboardInterrupt:
        print("Exiting...")

    # Cleanup
    zed.close()


if __name__ == "__main__":
    main()
