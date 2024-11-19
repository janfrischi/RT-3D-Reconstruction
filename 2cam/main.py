import numpy as np
from object_detection_and_point_cloud_processing import MainApp

if __name__ == "__main__":
    # Initialize the YOLO model and the camera system
    model_path = "models/yolo11l-seg.pt"
    sn_cam1 = 33137761
    sn_cam2 = 36829049

    # Color map and class_names for the classes
    color_map = {
        0: [15, 82, 186],  # Person - sapphire
        39: [255, 255, 0],  # Bottle - yellow
        41: [63, 224, 208],  # Cup - turquoise
        62: [255, 0, 255],  # Laptop - magenta
        64: [0, 0, 128],  # Mouse - navy
        66: [255, 0, 0],  # Keyboard - red
        73: [0, 255, 0]  # Book - green
    }
    class_names = {
        0: "Person",
        39: "Bottle",
        41: "Cup",
        62: "Laptop",
        64: "Mouse",
        66: "Keyboard",
        73: "Book"
    }
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

    # Initialize the parameters for both cameras
    init_params1 = {
        "resolution": "HD720",
        "fps": 60,
        "depth_mode": "NEURAL",
        "min_distance": 0.4,
        "units": "METER"
    }
    init_params2 = {
        "resolution": "HD720",
        "fps": 60,
        "depth_mode": "NEURAL",
        "min_distance": 0.4,
        "units": "METER"
    }

    # Create an instance of the MainApp class and run the application
    app = MainApp(model_path, sn_cam1, sn_cam2, color_map, class_names, T_chess_cam1, T_chess_cam2, T_robot_chess, init_params1, init_params2)
    for fused_pc in app.run():
        pass


