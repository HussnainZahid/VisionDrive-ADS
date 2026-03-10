"""
Utils – Tesla-style HUD Dashboard  (v3 – Professional Grade)
=============================================================
Improvements:
  • Full analogue speed gauge with tick marks and needle
  • Driving-state colour ring around gauge
  • TL colour indicator (large coloured dot)
  • Confidence bar for lane detection
  • Closest-object distance tape
  • Urgency banner (full-width flash on EMERGENCY)
  • PID / control telemetry panel
  • Smoothly-coloured lane fill (green/yellow/red confidence)
  • All drawing via OpenCV only (no pygame dependency)
"""

import cv2
import numpy as np
import math
import time
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.lane_detector   import LaneData
from perception.object_detector import DetectedObject
from decision.behavior_engine   import DrivingDecision, DriveState
from control.vehicle_controller import VehicleCommand
from config.settings            import FRAME_WIDTH, FRAME_HEIGHT

# ─── Palette ─────────────────────────────────────────────────────────────────
P = {
    "bg":        (10,  12,  18),
    "panel":     (18,  22,  32),
    "border":    (45,  55,  75),
    "accent":    (0,  210, 130),
    "text":      (215, 228, 245),
    "dim":       (95, 115, 145),
    "green":     (50, 210,  50),
    "yellow":    (0,  210, 210),
    "orange":    (0,  140, 255),
    "red":       (30,  30, 220),
    "white":     (255,255,255),
}

_URGENCY_COL = {
    "CLEAR":     (50, 210,  50),
    "LOW":       (50, 200, 210),
    "MEDIUM":    (20, 170, 255),
    "HIGH":      (0,  110, 255),
    "EMERGENCY": (0,   20, 255),
}

_STATE_LABEL = {
    DriveState.CRUISE:          ("CRUISE",    P["green"]),
    DriveState.SLOW_DOWN:       ("SLOW",      P["yellow"]),
    DriveState.EMERGENCY_BRAKE: ("E-STOP",    P["red"]),
    DriveState.EMERGENCY_CLEAR: ("RECOVERY",  P["orange"]),
    DriveState.TL_APPROACH:     ("TL APPR",   P["yellow"]),
    DriveState.TL_STOP:         ("TL STOP",   P["red"]),
    DriveState.TL_WAIT:         ("TL WAIT",   P["red"]),
    DriveState.TL_DEPART:       ("TL GO",     P["green"]),
    DriveState.SS_APPROACH:     ("SS APPR",   P["yellow"]),
    DriveState.SS_CREEP:        ("SS CREEP",  P["yellow"]),
    DriveState.SS_STOP:         ("SS STOP",   P["red"]),
    DriveState.SS_WAIT:         ("SS WAIT",   P["red"]),
    DriveState.SS_DEPART:       ("SS GO",     P["green"]),
}


def _txt(img, text, pos, scale=0.50, col=None, thick=1,
         font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, pos, font, scale, col or P["text"],
                thick, cv2.LINE_AA)


def _dark_overlay(img, x, y, w, h, alpha=0.72):
    """Dark semi-transparent rectangle."""
    roi  = img[y:y+h, x:x+w]
    dark = np.full_like(roi, P["bg"])
    cv2.addWeighted(dark, alpha, roi, 1-alpha, 0, roi)
    img[y:y+h, x:x+w] = roi


class HUDDashboard:
    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.W = width
        self.H = height
        self._start = time.time()

    def render(self,
               frame:        np.ndarray,
               lane:         LaneData,
               objects:      List[DetectedObject],
               decision:     DrivingDecision,
               cmd:          VehicleCommand,
               fps:          float,
               lane_detector=None,
               obj_detector=None,
               ) -> np.ndarray:

        out = cv2.resize(frame, (self.W, self.H))

        # 1. Lane overlay
        if lane_detector:
            out = lane_detector.draw_lanes(out, lane)

        # 2. Bounding boxes
        if obj_detector:
            out = obj_detector.draw_detections(out, objects)

        # 3. Top bar
        self._top_bar(out, decision, cmd, fps, len(objects), lane)

        # 4. Speed gauge (top-right)
        self._speed_gauge(out, cmd.speed_kmh, decision.state)

        # 5. Bottom telemetry strip
        self._bottom_strip(out, lane, decision, cmd)

        # 6. Traffic-light indicator
        if decision.tl_state not in ("UNKNOWN", ""):
            self._tl_indicator(out, decision.tl_state)

        # 7. Steering bar
        self._steering_bar(out, cmd.steering)

        # 8. Edge map inset
        if lane.debug_edges is not None:
            self._edge_inset(out, lane.debug_edges)

        # 9. Urgency side strip + emergency flash
        self._urgency_strip(out, decision.urgency)
        if decision.urgency == "EMERGENCY":
            self._emergency_flash(out)

        # 10. Closest object tape
        if decision.closest_dist_m < 80:
            self._dist_tape(out, decision.closest_dist_m)

        return out

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _top_bar(self, img, dec, cmd, fps, n, lane):
        bh = 46
        _dark_overlay(img, 0, 0, self.W, bh, alpha=0.78)

        # System title
        _txt(img, "VISION-ADS", (14, 30), scale=0.72, col=P["accent"], thick=2)

        # State pill
        label, col = _STATE_LABEL.get(dec.state, (dec.state, P["dim"]))
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        px, py = 195, 10
        cv2.rectangle(img, (px,py), (px+tw+18, py+28), col, -1, cv2.LINE_AA)
        _txt(img, label, (px+9, py+21), scale=0.62, col=P["bg"], thick=2)

        # Lane confidence pill
        conf_col = P["green"] if lane.confidence > 0.7 else (
                   P["yellow"] if lane.confidence > 0.4 else P["orange"])
        conf_lbl = f"LANE {lane.confidence:.0%}"
        _txt(img, conf_lbl, (px+tw+35, 30), scale=0.48, col=conf_col)

        # Right side: FPS, detections
        _txt(img, f"FPS {fps:.1f}", (self.W-185, 30), scale=0.55, col=P["dim"])
        _txt(img, f"OBJ {n}", (self.W-110, 30), scale=0.55, col=P["dim"])

    # ── Speed gauge ───────────────────────────────────────────────────────────
    def _speed_gauge(self, img, speed, state):
        cx, cy = self.W - 85, 115
        ro, ri = 62, 52

        # Background disc
        cv2.circle(img, (cx,cy), ro, P["panel"], -1)
        cv2.circle(img, (cx,cy), ro, P["border"], 1, cv2.LINE_AA)

        # State ring colour
        _, ring_col = _STATE_LABEL.get(state, ("", P["accent"]))
        cv2.circle(img, (cx,cy), ro, ring_col, 3, cv2.LINE_AA)

        # Arc 210°→-30° (270° sweep), 0–120 km/h
        v_max = 120.0
        frac  = min(speed / v_max, 1.0)
        start_a = 210; sweep = 270
        arc_col = (P["green"] if speed < 45 else
                   P["yellow"] if speed < 80 else P["red"])

        # Background arc
        cv2.ellipse(img, (cx,cy), (ri,ri), 0,
                    -start_a, -(start_a-sweep), P["border"], 2)
        # Value arc
        if frac > 0:
            cv2.ellipse(img, (cx,cy), (ri,ri), 0,
                        -start_a, -int(start_a-sweep*frac), arc_col, 3)

        # Tick marks every 20 km/h
        for v in range(0, 121, 20):
            angle_deg = start_a - (v/v_max)*sweep
            angle_r   = math.radians(angle_deg)
            outer = ro - 4; inner = ro - 12
            x1 = int(cx + outer*math.cos(angle_r))
            y1 = int(cy - outer*math.sin(angle_r))
            x2 = int(cx + inner*math.cos(angle_r))
            y2 = int(cy - inner*math.sin(angle_r))
            cv2.line(img, (x1,y1), (x2,y2), P["dim"], 1, cv2.LINE_AA)

        # Needle
        needle_a = math.radians(start_a - frac*sweep)
        nx = int(cx + (ri-8)*math.cos(needle_a))
        ny = int(cy - (ri-8)*math.sin(needle_a))
        cv2.line(img, (cx,cy), (nx,ny), P["white"], 2, cv2.LINE_AA)
        cv2.circle(img, (cx,cy), 4, P["white"], -1)

        # Numeric
        spd_str = f"{speed:.0f}"
        (tw,th),_ = cv2.getTextSize(spd_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        _txt(img, spd_str, (cx-tw//2, cy+8), scale=0.85,
             col=arc_col, thick=2)
        _txt(img, "km/h", (cx-14, cy+24), scale=0.33, col=P["dim"])

    # ── Bottom telemetry strip ────────────────────────────────────────────────
    def _bottom_strip(self, img, lane, dec, cmd):
        y0 = self.H - 52
        _dark_overlay(img, 0, y0, self.W, 52, alpha=0.80)

        cols = [
            ("SPEED",    f"{cmd.speed_kmh:.1f} km/h"),
            ("TARGET",   f"{dec.target_speed:.1f} km/h"),
            ("OFFSET",   f"{lane.lane_center_offset:+.1f} px"),
            ("ANGLE",    f"{lane.lane_angle:+.1f}°"),
            ("CURV",     f"{lane.curvature:.5f}"),
            ("STEER",    f"{cmd.steering:+.4f}"),
            ("THROTTLE", f"{cmd.throttle:.3f}"),
            ("BRAKE",    f"{cmd.brake:.3f}"),
            ("CONF",     f"{lane.confidence:.2f}"),
            ("NEAREST",  f"{dec.closest_dist_m:.1f}m"),
            ("URGENCY",  dec.urgency),
        ]
        cw = self.W // len(cols)
        for i,(lbl,val) in enumerate(cols):
            x = i*cw + 6
            # Urgency column colour
            vc = _URGENCY_COL.get(val, P["text"]) if lbl=="URGENCY" else P["text"]
            _txt(img, lbl, (x, y0+16), scale=0.36, col=P["dim"])
            _txt(img, val, (x, y0+36), scale=0.44, col=vc)

    # ── Traffic-light indicator ────────────────────────────────────────────────
    def _tl_indicator(self, img, tl_state):
        col = {"RED":P["red"],"GREEN":P["green"],"YELLOW":P["yellow"]}.get(
              tl_state, P["dim"])
        cx, cy = self.W - 85, 200
        cv2.circle(img, (cx,cy), 18, P["panel"], -1)
        cv2.circle(img, (cx,cy), 18, P["border"], 1)
        cv2.circle(img, (cx,cy), 12, col, -1, cv2.LINE_AA)
        _txt(img, tl_state[:3], (cx-13, cy+22), scale=0.38, col=col)

    # ── Steering bar ──────────────────────────────────────────────────────────
    def _steering_bar(self, img, steer):
        bw, bh = 220, 10
        bx = (self.W-bw)//2
        by = self.H - 68
        _dark_overlay(img, bx-4, by-6, bw+8, bh+14, alpha=0.60)
        cv2.rectangle(img, (bx,by), (bx+bw,by+bh), P["border"], 1)
        cv2.line(img, (bx+bw//2, by-4), (bx+bw//2, by+bh+4), P["dim"], 1)
        nx = int(bx + bw//2 + steer*(bw//2))
        nx = max(bx+3, min(bx+bw-3, nx))
        sc = (P["green"] if abs(steer)<0.12 else
              P["yellow"] if abs(steer)<0.35 else P["orange"])
        cv2.rectangle(img, (nx-3,by-1), (nx+3,by+bh+1), sc, -1)
        _txt(img, f"STEER {steer:+.4f}", (bx, by-10), scale=0.36, col=P["dim"])

    # ── Edge map inset ────────────────────────────────────────────────────────
    def _edge_inset(self, img, edges):
        iw, ih = 168, 95
        pad = 8
        x0 = self.W - iw - pad
        y0 = self.H - ih - pad - 54
        rgb  = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(rgb, (iw,ih))
        tint  = np.zeros_like(thumb); tint[:,:,1] = 70
        cv2.addWeighted(thumb, 0.8, tint, 0.35, 0, thumb)
        cv2.rectangle(img, (x0-2,y0-2), (x0+iw+2,y0+ih+2), P["border"], 1)
        img[y0:y0+ih, x0:x0+iw] = thumb
        _txt(img, "EDGE MAP", (x0+2, y0+ih-3), scale=0.28, col=P["dim"])

    # ── Urgency side strip ────────────────────────────────────────────────────
    def _urgency_strip(self, img, urgency):
        col = _URGENCY_COL.get(urgency, P["border"])
        ht  = {"CLEAR":int(self.H*0.12),"LOW":int(self.H*0.25),
               "MEDIUM":int(self.H*0.50),"HIGH":int(self.H*0.75),
               "EMERGENCY":self.H}.get(urgency,0)
        if ht:
            cv2.rectangle(img, (0, self.H-ht), (5, self.H), col, -1)

    # ── Emergency flash overlay ───────────────────────────────────────────────
    def _emergency_flash(self, img):
        t = time.time()
        if int(t*4) % 2 == 0:   # 2 Hz flash
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (self.W,self.H), (0,0,180), 10)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            _txt(img, "⚠  EMERGENCY BRAKE  ⚠",
                 (self.W//2-180, 80), scale=1.0, col=(0,0,255), thick=3)

    # ── Closest-distance tape ─────────────────────────────────────────────────
    def _dist_tape(self, img, dist_m):
        """Vertical tape on right edge showing proximity."""
        tap_h = int(self.H * 0.60)
        x0 = self.W - 22
        y0 = int(self.H * 0.15)
        cv2.rectangle(img, (x0, y0), (x0+8, y0+tap_h), P["panel"], -1)
        # Filled portion: 0m=full, 80m=empty
        frac  = max(0.0, 1.0 - dist_m/80.0)
        fy    = int(tap_h * frac)
        col   = (P["red"] if dist_m<10 else
                 P["orange"] if dist_m<20 else
                 P["yellow"] if dist_m<35 else P["green"])
        if fy > 0:
            cv2.rectangle(img, (x0, y0+tap_h-fy), (x0+8, y0+tap_h), col, -1)
        _txt(img, f"{dist_m:.0f}m", (x0-14, y0+tap_h+14),
             scale=0.32, col=col)
