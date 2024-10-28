# Real-Time 3D Reconstruction and Object Tracking Pipeline

This project implements a real-time 3D reconstruction pipeline using a ZED stereo camera, YOLOv11 object detection and segmentation, and Open3D for 3D visualization. The goal is to segment and track objects in real-time, creating 3D point clouds with depth information obtained from the ZED camera.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

## Overview

The pipeline processes video frames from a ZED stereo camera, performs real-time object detection and segmentation using YOLOv11, and combines the detected object masks with depth data from the ZED camera to generate 3D point clouds. Using Open3D, these point clouds are visualized and updated dynamically to allow real-time 3D reconstruction.

### Key Components

- **ZED Stereo Camera**: Provides video frames and depth information.
- **YOLOv11**: Detects and segments objects in the video frames.
- **Open3D**: Visualizes 3D point clouds generated from segmented masks.

## Features

- **Real-Time Object Detection and Tracking**: Uses YOLOv11 with ByteTrack to detect and track objects within video frames.
- **3D Point Cloud Generation**: Projects segmented 2D masks to 3D space using depth information from the ZED camera.
- **Interactive Visualization**: Displays 3D point clouds in an Open3D window with options for real-time updates and static scene capture.
- **Customizable Color Map**: Assigns distinct colors to different object classes for easier visualization.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:janfrischi/RT-3D-Reconstruction.git
   cd RT-3D-Reconstruction
