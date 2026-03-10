#!/usr/bin/env python3
"""
Vision-Only Autonomous Driving System  v2
==========================================
Performance optimisations vs v1:
  • Detection frame-skip (--det-skip N): run YOLO every N frames,
    reuse results in between → 2-3× FPS boost on CPU
  • numpy RankWarning suppressed globally (before any import)
  • Curvature uses geometric estimate (no polyfit)
  • frame_idx replaces source.position for telemetry (avoids extra CAP call)

Press  Q  or  ESC  to quit.
"""

import warnings
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import sys
import os
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from perception.lane_detector   import LaneDetector
from perception.object_detector import ObjectDetector
from decision.behavior_engine   import BehaviorEngine
from control.vehicle_controller import VehicleController
from utils.hud                  import HUDDashboard
from utils.video_io             import VideoSource, VideoRecorder
from utils.perf_monitor         import PerfMonitor
from config.settings            import (TARGET_FPS, RECORDING, INPUT,
                                        FRAME_WIDTH, FRAME_HEIGHT)


def parse_args():
    p = argparse.ArgumentParser(description="Vision ADS")
    p.add_argument("--source",   type=str, default=INPUT["source"])
    p.add_argument("--no-record",action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--width",    type=int, default=FRAME_WIDTH)
    p.add_argument("--height",   type=int, default=FRAME_HEIGHT)
    p.add_argument("--det-skip", type=int, default=2,
                   help="Run object detection every N frames (default: 2). "
                        "Higher = faster. 1 = every frame.")
    return p.parse_args()


def run(args):
    print("=" * 60)
    print("  VISION-ONLY AUTONOMOUS DRIVING SYSTEM  v2")
    print("  Press Q or ESC to quit")
    print("=" * 60)

    print("\n[INIT] Loading modules...")
    lane_detector   = LaneDetector(args.width, args.height)
    obj_detector    = ObjectDetector()
    behavior_engine = BehaviorEngine()
    controller      = VehicleController()
    hud             = HUDDashboard(args.width, args.height)
    perf            = PerfMonitor()

    det_skip = max(1, args.det_skip)
    print(f"[INIT] Detection frame-skip = {det_skip}  "
          f"(YOLO runs every {det_skip} frame{'s' if det_skip > 1 else ''})")

    try:
        src_arg = int(args.source)
    except (ValueError, TypeError):
        src_arg = args.source

    source = VideoSource(src_arg, loop=True)

    recorder = None
    if not args.no_record and RECORDING["enabled"]:
        recorder = VideoRecorder(
            path=RECORDING["output_path"],
            fps=RECORDING["fps"],
            width=args.width,
            height=args.height,
        )

    if not args.headless:
        cv2.namedWindow("Vision ADS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vision ADS", args.width, args.height)

    frame_interval = 1.0 / TARGET_FPS
    dt             = frame_interval
    prev_time      = time.time()
    frame_idx      = 0
    objects        = []

    print("\n[RUN] Pipeline started.\n")

    while True:
        loop_start = time.time()
        perf.tick()

        # 1. Read
        frame = source.read()
        if frame is None:
            print("[INFO] End of video stream.")
            break
        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height),
                               interpolation=cv2.INTER_LINEAR)

        # 2. Lane detection
        perf.start("lane")
        lane_data = lane_detector.detect(frame)
        perf.stop("lane")

        # 3. Object detection (frame-skipped)
        if frame_idx % det_skip == 0:
            perf.start("detect")
            left_x  = (lane_data.left_lane_points[0][0]
                       if len(lane_data.left_lane_points) == 2 else None)
            right_x = (lane_data.right_lane_points[0][0]
                       if len(lane_data.right_lane_points) == 2 else None)
            objects = obj_detector.detect(frame, left_x, right_x)
            perf.stop("detect")

        # 4. Decision
        decision = behavior_engine.plan(
            lane_data, objects, controller.current_speed_kmh)

        # 5. Control
        command = controller.step(lane_data, decision, dt)

        # 6. HUD
        perf.start("hud")
        output = hud.render(
            frame, lane_data, objects, decision, command,
            fps=perf.fps(),
            lane_detector=lane_detector,
            obj_detector=obj_detector,
        )
        perf.stop("hud")

        # 7. Record
        if recorder:
            if not recorder.write(output):
                print(f"[RECORD] Clip saved -> {RECORDING['output_path']}")
                recorder.release()
                recorder = None

        # 8. Display
        if not args.headless:
            cv2.imshow("Vision ADS", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

        # 9. Telemetry every 60 frames
        if frame_idx % 60 == 0:
            s = perf.stats()
            print(
                f"  Frame {frame_idx:5d} | FPS:{s['fps']:5.1f} | "
                f"Lane:{s['lane_ms']:4.1f}ms | Det:{s['detect_ms']:5.1f}ms | "
                f"Speed:{command.speed_kmh:5.1f}km/h | "
                f"State:{decision.state:<20s} | Urgency:{decision.urgency}"
            )

        # Timing
        elapsed = time.time() - loop_start
        sleep_t = max(0.0, frame_interval - elapsed)
        if sleep_t > 0:
            time.sleep(sleep_t)
        dt        = time.time() - prev_time
        prev_time = time.time()
        frame_idx += 1

    # Cleanup
    source.release()
    if recorder:
        recorder.release()
    if not args.headless:
        cv2.destroyAllWindows()

    print("\n[DONE] Vision ADS stopped.")
    s = perf.stats()
    print(f"  Average FPS   : {s['fps']:.1f}")
    print(f"  Lane latency  : {s['lane_ms']:.1f} ms")
    print(f"  Detect latency: {s['detect_ms']:.1f} ms (on frames where detection ran)")
    print(f"  HUD latency   : {s['hud_ms']:.1f} ms")


if __name__ == "__main__":
    args = parse_args()
    run(args)
