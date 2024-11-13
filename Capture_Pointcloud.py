import open3d as o3d
import numpy as np
import pyzed.sl as sl


def main():
    print("Initializing ZED camera for point cloud visualization...")

    # Set up ZED camera parameters
    init_params = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.ULTRA,  # High-quality depth
        coordinate_units=sl.UNIT.METER,  # Meters as units
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL-style coordinates
    )

    # Open the ZED camera
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", repr(status))
        exit()

    # Set the resolution for capturing
    resolution = sl.Resolution(720, 404)

    # Allocate memory for point cloud retrieval
    zed_point_cloud = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # Initialize Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("ZED Point Cloud Viewer")

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Define a callback function to save the point cloud as a .ply file
    def save_point_cloud(vis):
        if len(pcd.points) > 0:
            o3d.io.write_point_cloud("zed_point_cloud.ply", pcd)
            print("Point cloud saved as zed_point_cloud.ply")
        else:
            print("No points to save.")
        return False

    # Register callback to save point cloud on pressing "S"
    vis.register_key_callback(ord("S"), save_point_cloud)

    # Main loop for capturing and visualizing the point cloud
    try:
        while True:
            # Capture a new frame from the ZED camera
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve and process the point cloud
                zed.retrieve_measure(zed_point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)
                point_cloud_data = zed_point_cloud.get_data()

                # Extract XYZ and color information from the point cloud data
                xyz = point_cloud_data[:, :, :3].reshape(-1, 3)  # XYZ coordinates
                rgba = point_cloud_data[:, :, 3].reshape(-1)  # RGBA color data

                # Filter out invalid points (NaNs and zero-length vectors)
                mask = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0)
                valid_xyz = xyz[mask]
                valid_colors = rgba[mask]

                if valid_xyz.shape[0] > 0:
                    # Set points and colors for Open3D visualization
                    pcd.points = o3d.utility.Vector3dVector(valid_xyz)
                    colors = np.c_[valid_colors, valid_colors, valid_colors] / 255.0  # Normalize to [0, 1]
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                    # Update Open3D visualization
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                else:
                    print("No valid points to display in the current frame.")

                # Check if the window is closed
                if not vis.poll_events():
                    break
    finally:
        # Clean up
        vis.destroy_window()
        zed.close()
        print("Visualization closed and ZED camera released.")


if __name__ == "__main__":
    main()
