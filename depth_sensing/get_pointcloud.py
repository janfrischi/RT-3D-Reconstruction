import sys
from ogl_viewer import viewer as gl
import pyzed.sl as sl
import argparse


def main():
    print("Running Live Point Cloud Viewer... Press 'Esc' to quit\nPress 's' to save the point cloud.")

    # Initialize the ZED camera
    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.ULTRA,  # High accuracy depth mode
        coordinate_units=sl.UNIT.METER,
        camera_resolution = sl.RESOLUTION.HD720,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # Standard OpenGL coordinate system
    )
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Set resolution for the point cloud
    resolution = sl.Resolution(720, 404)  # Example resolution, adjust as needed

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
