import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl

def main():
    print("Running Live Point Cloud Viewer for Two Cameras... Press 'Esc' to quit.")

    # Initialize the ZED camera objects
    zed1 = sl.Camera()
    zed2 = sl.Camera()

    # Serial numbers for the cameras
    sn_cam1 = 33137761  # Replace with your first camera's serial number
    sn_cam2 = 36829049  # Replace with your second camera's serial number

    # Set initialization parameters for camera 1
    init_params1 = sl.InitParameters()
    init_params1.set_from_serial_number(sn_cam1)
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params1.depth_minimum_distance = 0.4
    init_params1.coordinate_units = sl.UNIT.METER

    # Set initialization parameters for camera 2
    init_params2 = sl.InitParameters()
    init_params2.set_from_serial_number(sn_cam2)
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 30
    init_params2.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params2.depth_minimum_distance = 0.4
    init_params2.coordinate_units = sl.UNIT.METER

    # Open the cameras
    if zed1.open(init_params1) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera 1")
        return
    if zed2.open(init_params2) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera 2")
        zed1.close()
        return

    # Set resolution for the point clouds
    resolution = sl.Resolution(720, 404)  # Adjust as needed

    # Get the camera models
    camera_model1 = zed1.get_camera_information().camera_model
    camera_model2 = zed2.get_camera_information().camera_model

    # Initialize the OpenGL viewer for two cameras
    viewer = gl.GLViewer()
    viewer.init(2, sys.argv, [camera_model1, camera_model2], resolution)

    # Create point cloud objects for each camera
    point_cloud1 = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud2 = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    while viewer.is_available():
        # Grab frames from both cameras
        if zed1.grab() == sl.ERROR_CODE.SUCCESS:
            zed1.retrieve_measure(point_cloud1, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)
            viewer.updateData(0, point_cloud1)  # Update viewer with camera 1 data
        if zed2.grab() == sl.ERROR_CODE.SUCCESS:
            zed2.retrieve_measure(point_cloud2, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)
            viewer.updateData(1, point_cloud2)  # Update viewer with camera 2 data

    # Cleanup
    viewer.exit()
    zed1.close()
    zed2.close()

if __name__ == "__main__":
    main()
