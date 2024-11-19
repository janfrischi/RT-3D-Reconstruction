import pyzed.sl as sl
import cv2
import time
import torch
from ultralytics import YOLO

# This script demonstrates how to perform object tracking with YOLO11 segmentation model on ZED camera feed
# No point cloud processing is done in this script

def main():

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pretrained YOLO11 model and move it to the device
    model = YOLO("yolo11l-seg.pt").to(device)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set camera resolution
    init_params.camera_fps = 60  # Set camera FPS
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Enable depth sensing - mode can be PERFORMANCE, MEDIUM, QUALITY
    init_params.depth_minimum_distance = 0.3
    init_params.coordinate_units = sl.UNIT.METER  # Set units to meters

    # Check if the camera is opened successfully
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        exit(1)

    # Get the camera calibration parameters
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    calibration_params_left = calibration_params.left_cam
    calibration_params_right = calibration_params.right_cam
    # Translation between left and right eye on x-axis of the camera
    tx = calibration_params.stereo_transform.get_translation().get()[0]
    print(f"Left camera calibration parameters: {calibration_params_left.fx}, {calibration_params_left.fy}, {calibration_params_left.cx}, {calibration_params_left.cy}")
    print(f"Right camera calibration parameters: {calibration_params_right.fx}, {calibration_params_right.fy}, {calibration_params_right.cx}, {calibration_params_right.cy}")
    print(f"Translation between left and right eye on x-axis: {tx.round(5)}")

    # Create Mat object to hold the image
    image = sl.Mat()
    depth = sl.Mat()
    key = ''
    print("Press 'q' to quit the video feed.")

    # Create named window

    fps_values = []
    while key != ord('q'):
        start_time = time.time()  # Start time for FPS calculation

        # Grab an image from the camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = image.get_data()

            # Convert frame from 4 channels (RGBA) to 3 channels (RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Perform object tracking with yolo11-seg model
            results = model.track(source=frame,
                                  imgsz= 640, # Resize input frame to 640x640 for processing
                                  max_det=20,
                                  classes=[0,39,41,64,66,73],
                                  conf= 0.5,
                                  device=device,
                                  half=True,
                                  persist=True,
                                  retina_masks=True,
                                  tracker="ultralytics/cfg/trackers/bytetrack.yaml")

            # Visualize the results on the frame, results is a class for storing the results of the detection
            annotated_frame = results[0].plot(line_width=2, font_size=18)

            # Display the distance for each detected object
            for box in results[0].boxes:
                # Get the center of the bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get the depth value at the center of the bounding box
                depth_value = depth.get_value(center_x, center_y)[1]
                if not depth_value:
                    distance_text = "N/A"
                else:
                    distance_text = f"{depth_value:.2f} m"

                # Display the distance on the frame
                cv2.putText(
                    annotated_frame,
                    distance_text,
                    (center_x - 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )


            # Check for user input
            key = cv2.waitKey(1)

            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_values.append(fps)

            # Calculate a moving average to smooth the FPS display, more stable display of the fps
            if len(fps_values) > 10:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame)


    # Close the camera and exit
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()