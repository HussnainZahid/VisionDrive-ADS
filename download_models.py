#!/usr/bin/env python3
"""
Download YOLOv4-tiny and MobileNet SSD weights.
Run once before main.py:
    python download_models.py
"""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = [
    # YOLOv4-tiny
    (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "yolov4-tiny.cfg",
    ),
    (
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "yolov4-tiny.weights",
    ),
    # COCO names
    (
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "coco.names",
    ),
    # MobileNet SSD (Caffe) – much smaller, good fallback
    (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        "MobileNetSSD_deploy.prototxt",
    ),
    (
        "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc",
        "MobileNetSSD_deploy.caffemodel",  # ~23 MB
    ),
]


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  [SKIP] {os.path.basename(dest)} already exists.")
        return
    print(f"  [DL]   {os.path.basename(dest)}  ← {url[:70]}…")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"         ✓ saved ({os.path.getsize(dest)//1024} KB)")
    except Exception as e:
        print(f"         ✗ FAILED: {e}")


if __name__ == "__main__":
    print("Downloading model weights…\n")
    for url, name in FILES:
        download(url, os.path.join(MODELS_DIR, name))
    print("\nDone. Run: python main.py")
