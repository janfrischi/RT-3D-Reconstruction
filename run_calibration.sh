#!/bin/bash

# Define the project directory
PROJECT_DIR="/home/janfrischknecht/Documents/Semesterprojekt/Camera_Calibration/camera_calibration"

# Step 1: Run multi_camera.py and stop execution after 4 seconds
echo "Running multi_camera.py..."
python3 "$PROJECT_DIR/multi_camera.py" &
MULTI_CAMERA_PID=$!
sleep 15
kill $MULTI_CAMERA_PID
echo "multi_camera.py execution stopped."

# Step 2: Run svo_extract.py
echo "Running svo_extract.py..."
python3 "$PROJECT_DIR/svo_extract.py"
echo "svo_extract.py completed."

# Step 3: Create a 'calib' folder inside Rec_1 and copy recorded files
echo "Setting up calibration folder..."
RECORDING_DIR="$PROJECT_DIR/Rec_1"
CALIB_DIR="$RECORDING_DIR/calib"
mkdir -p "$CALIB_DIR"
cp "$RECORDING_DIR/Rec/"* "$CALIB_DIR"
echo "Files copied to calib folder."

# Step 4: Run extrinsic_calib.py
echo "Running extrinsic_calib.py..."
python3 "$PROJECT_DIR/extrinsic_calib.py"
echo "extrinsic_calib.py completed."

# Step 5: Run write_to_human_readable_file.py
echo "Running write_to_human_readable_file.py..."
python3 "$PROJECT_DIR/write_to_human_readable_file.py"
echo "write_to_human_readable_file.py completed."

echo "Workflow completed successfully."
