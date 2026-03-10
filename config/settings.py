"""
Vision ADS – Global Configuration  (v3)
========================================
All tuneable parameters.  Change here; no code edits required.
"""

# ─── Camera & Display ────────────────────────────────────────────────────────
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
TARGET_FPS    = 20

# ─── Lane Detection ──────────────────────────────────────────────────────────
LANE = {
    "gaussian_blur_kernel": (5, 5),
    # Canny thresholds used as floor/ceiling; actual thresholds are Otsu-derived
    "canny_low":   40,
    "canny_high": 180,
    # Hough (tight pass)
    "hough_rho":             1,
    "hough_theta_deg":       1,
    "hough_threshold":       38,
    "hough_min_line_length": 35,
    "hough_max_line_gap":    22,
    # ROI
    "roi_top_ratio":    0.57,
    "roi_bottom_ratio": 0.97,
    # Slope filter
    "lane_slope_min": 0.38,
    "lane_slope_max": 3.80,
    # EMA smoothing  (legacy – actual values in LaneDetector __init__)
    "smoothing_alpha": 0.20,
    # Curvature speed scale (km/h per unit curvature × 2000)
}

# ─── Object Detection ────────────────────────────────────────────────────────
DETECTION = {
    "confidence_threshold": 0.45,   # global default; per-class values in object_detector.py
    "nms_threshold":        0.38,
    "input_size":           (416, 416),
    # YOLOv4-tiny paths (relative to project root)
    "yolo_cfg":      "models/yolov4-tiny.cfg",
    "yolo_weights":  "models/yolov4-tiny.weights",
    "coco_names":    "models/coco.names",
    # MobileNet SSD paths
    "mobilenet_proto":  "models/MobileNetSSD_deploy.prototxt",
    "mobilenet_model":  "models/MobileNetSSD_deploy.caffemodel",
    # Driving-relevant COCO classes
    "target_classes": {
        "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
        "bus": 5, "truck": 7, "traffic light": 9, "stop sign": 11,
    },
}

# ─── Monocular Distance ───────────────────────────────────────────────────────
# Calibrate FOCAL_LENGTH_PX using a known object at a known distance:
#   focal = (measured_bbox_height_px * real_distance_m) / real_object_height_m
FOCAL_LENGTH_PX = 900

REAL_HEIGHTS_M = {
    "person":       1.75,
    "car":          1.50,
    "truck":        3.80,
    "bus":          3.20,
    "motorcycle":   1.15,
    "bicycle":      1.05,
    "traffic light": 0.70,
    "stop sign":    0.75,
}

# ─── Safety Thresholds ───────────────────────────────────────────────────────
SAFETY = {
    "emergency_person_m":   5.5,
    "emergency_vehicle_m":  9.0,
    "high_m":              18.0,
    "medium_m":            32.0,
    "corridor_width_frac":  0.38,
}

# ─── Speed & Control ─────────────────────────────────────────────────────────
CONTROL = {
    "target_speed_kmh":      35.0,
    "max_throttle":           0.80,
    "max_brake":              1.0,
    # Steering PID
    "steering_kp":            0.010,
    "steering_kd":            0.004,
    "steering_smooth":        0.30,
    # Speed PID
    "speed_kp":               0.06,
    # Curvature
    "curvature_speed_scale":  20.0,
    # Traffic light / stop sign
    "stop_sign_wait_sec":     3.0,
    "tl_stop_distance_m":    16.0,
    "tl_slow_distance_m":    28.0,
}

# ─── Visualisation ───────────────────────────────────────────────────────────
VIZ = {
    "bbox_colors": {
        "person":        (0,  70, 255),
        "car":           (0, 200, 255),
        "truck":         (0, 130, 255),
        "bus":           (20, 170, 255),
        "motorcycle":    (190, 40, 255),
        "bicycle":       (110, 40, 210),
        "traffic light": (0,  255, 180),
        "stop sign":     (0,  40, 255),
        "default":       (180,180,180),
    },
    "font_scale": 0.50,
    "font_thick": 1,
}

# ─── Recording ───────────────────────────────────────────────────────────────
RECORDING = {
    "enabled":      True,
    "output_path":  "data/output_recording.mp4",
    "codec":        "mp4v",
    "duration_sec": 60,
    "fps":          20,
}

# ─── Input ───────────────────────────────────────────────────────────────────
INPUT = {
    "source": "data/drive_video.mp4",
    "loop":   True,
}
