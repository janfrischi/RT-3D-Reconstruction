import sys
from ogl_viewer import viewer as gl
import pyzed.sl as sl
import numpy as np
import argparse

T_chess_cam1 = np.array([[0.6653, 0.4827, -0.5696, 0.5868],
                             [-0.7466, 0.4314, -0.5065, 0.7718],
                             [0.0012, 0.7622, 0.6473, -0.7245],
                             [0.0000, 0.0000, 0.0000, 1.0000]])

T_chess_cam2 = np.array([[0.3981, -0.6302, 0.6666, -0.5739],
                             [0.9173, 0.2688, -0.2937, 0.3581],
                             [0.0059, 0.7284, 0.6851, -0.6835],
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


def main():
    print("Running Live Point Cloud Viewer... Press 'Esc' to quit\nPress 's' to save the point cloud.")

    # Initialize the ZED camera
    sn_cam1 = 33137761
    sn_cam2 = 36829049

    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Convention for OpenGL
    init_params1.coordinate_units = sl.UNIT.METER


    zed = sl.Camera()

    status = zed.open(init_params1)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Set resolution for the point cloud
    resolution = sl.Resolution(1280, 720)  # Example resolution, adjust as needed

    # Get the camera model
    camera_model = zed.get_camera_information().camera_model

    # Initialize the OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, camera_model, resolution)

    # Create a point cloud object to hold data
    point_cloud = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)



    while viewer.is_available():
        # Grab frames from the ZED camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the point cloud with color
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)

            # Convert the point cloud to a NumPy array
            # Get the raw data
            point_cloud_np = point_cloud.get_data()[:, :, :3]
            # Reshape to (N, 3), where N is the number of points
            point_cloud_np = point_cloud_np.reshape(-1, 3)
            # Remove invalid points (NaN or Inf values)
            valid_mask = ~np.isnan(point_cloud_np).any(axis=1) & ~np.isinf(point_cloud_np).any(axis=1)
            point_cloud_cam1 = point_cloud_np[valid_mask]

            # Transform the point cloud to the robot base frame
            # Apply the rotation matrix to the point cloud
            point_cloud_cam1_transformed = np.dot(rotation_robot_cam1, point_cloud_cam1.T).T + origin_cam1

            print(f"Dimension of point cloud: {point_cloud_np.shape}")


            # Update the OpenGL viewer with the new point cloud data
            viewer.updateData(point_cloud)

            # Save point cloud if requested
            if viewer.save_data:
                point_cloud_to_save = sl.Mat()
                zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                save_status = point_cloud_to_save.write("Pointcloud.ply")
                if save_status == sl.ERROR_CODE.SUCCESS:
                    print("Point cloud saved as 'Pointcloud.ply'.")
                else:
                    print("Failed to save point cloud.")
                viewer.save_data = False

    # Cleanup
    viewer.exit()
    zed.close()


if __name__ == "__main__":
    main()
