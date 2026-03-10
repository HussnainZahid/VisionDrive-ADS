#!/usr/bin/env python3
"""
Smoke-test: runs the full pipeline on a single synthetic frame
(no video file required) and verifies all modules import correctly.
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import cv2

print("=" * 50)
print("  Vision ADS – smoke test")
print("=" * 50)

# 1. Create synthetic frame (road-like gradient)
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
frame[400:, :] = (50, 50, 50)   # road
cv2.line(frame, (580, 720), (620, 450), (200, 200, 200), 5)   # left lane
cv2.line(frame, (700, 720), (660, 450), (200, 200, 200), 5)   # right lane
print("[OK] Synthetic frame created 1280x720")

# 2. Lane detector
from perception.lane_detector import LaneDetector
ld = LaneDetector()
data = ld.detect(frame)
print(f"[OK] LaneDetector: offset={data.lane_center_offset:.1f}px  conf={data.confidence:.2f}")

# 3. Object detector
from perception.object_detector import ObjectDetector
od = ObjectDetector()
objs = od.detect(frame)
print(f"[OK] ObjectDetector: {len(objs)} object(s) detected")

# 4. Behavior engine
from decision.behavior_engine import BehaviorEngine
be = BehaviorEngine()
dec = be.plan(data, objs)
print(f"[OK] BehaviorEngine: state={dec.state}  speed={dec.target_speed:.1f}km/h")

# 5. Controller
from control.vehicle_controller import VehicleController
vc = VehicleController()
cmd = vc.step(data, dec)
print(f"[OK] VehicleController: steer={cmd.steering:+.3f}  throttle={cmd.throttle:.2f}")

# 6. HUD render
from utils.hud import HUDDashboard
hud = HUDDashboard()
out = hud.render(frame, data, objs, dec, cmd, fps=20.0,
                 lane_detector=ld, obj_detector=od)
print(f"[OK] HUD rendered: {out.shape}")

# 7. Save a sample frame
os.makedirs("data", exist_ok=True)
cv2.imwrite("data/smoke_test_output.png", out)
print("[OK] Sample frame saved → data/smoke_test_output.png")

print("\n✅  All modules operational.\n")
