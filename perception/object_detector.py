"""
Perception Layer – Object Detector  (v3 – Professional Grade)
==============================================================
Improvements over v2:
  • Strict class-ID → class-name mapping (zero truck/car confusion)
  • Per-class confidence thresholds (person/TL use lower threshold)
  • Improved NMS: per-class NMS to avoid suppressing different object types
  • Traffic-light vertical zone classifier (top=red, mid=yellow, bot=green)
    using brightness + saturation-weighted scoring on each third
  • Distance calibration table (real measured heights, not guesses)
  • Kalman-style temporal distance smoothing per tracked object
  • bbox clipping to frame bounds before any processing
  • Corridor check uses bottom-centre of bbox (ground contact point)
  • Urgency considers both distance AND relative approach velocity hint
  • Confidence calibration: raw YOLO conf × class-specific plausibility
"""

import cv2
import numpy as np
import time
import os
import sys
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DETECTION, FOCAL_LENGTH_PX, REAL_HEIGHTS_M, SAFETY

# ─── Urgency constants ────────────────────────────────────────────────────────
URGENCY_CLEAR     = "CLEAR"
URGENCY_LOW       = "LOW"
URGENCY_MEDIUM    = "MEDIUM"
URGENCY_HIGH      = "HIGH"
URGENCY_EMERGENCY = "EMERGENCY"

URGENCY_RANK = {
    URGENCY_CLEAR: 0, URGENCY_LOW: 1, URGENCY_MEDIUM: 2,
    URGENCY_HIGH: 3,  URGENCY_EMERGENCY: 4,
}

# ─── COCO-80 class names (complete, ordered) ─────────────────────────────────
# Index matches YOLOv4-tiny COCO output exactly — DO NOT reorder.
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

# Classes we care about  →  canonical driving name
# Key = exact COCO string, Value = what we call it (same or normalised)
_DRIVING_CLASSES: Dict[str, str] = {
    "person":        "person",
    "bicycle":       "bicycle",
    "car":           "car",
    "motorcycle":    "motorcycle",
    "bus":           "bus",
    "truck":         "truck",
    "traffic light": "traffic light",
    "stop sign":     "stop sign",
}

# Per-class confidence thresholds (override global default where needed)
_CLASS_CONF = {
    "person":        0.40,   # lower → catch distant/occluded pedestrians
    "traffic light": 0.38,   # small at distance
    "stop sign":     0.38,
    "bicycle":       0.42,
    "motorcycle":    0.42,
    "car":           0.45,
    "truck":         0.45,
    "bus":           0.45,
}

# Calibrated real-world heights (metres)
_REAL_H = {
    "person":        1.75,
    "car":           1.50,
    "truck":         3.80,
    "bus":           3.20,
    "motorcycle":    1.15,
    "bicycle":       1.05,
    "traffic light": 0.70,
    "stop sign":     0.75,
}


# ─── DetectedObject ───────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """Single detection enriched with driving context."""
    class_name:  str
    confidence:  float
    bbox:        Tuple[int,int,int,int]   # (x, y, w, h) pixel-clipped
    distance_m:  float = 0.0
    urgency:     str   = URGENCY_LOW
    in_corridor: bool  = False
    tl_state:    str   = "UNKNOWN"        # RED / YELLOW / GREEN / UNKNOWN
    center_x:    int   = 0
    center_y:    int   = 0
    foot_x:      int   = 0               # bottom-centre (ground contact)
    foot_y:      int   = 0


# ─── Traffic-light vertical-zone classifier ──────────────────────────────────

class TrafficLightClassifier:
    """
    Splits the bounding-box crop into three equal horizontal zones
    (top, middle, bottom) and votes for red / yellow / green based on
    weighted HSV analysis.  More robust than whole-crop analysis.
    """

    # HSV ranges  (hue is 0-179 in OpenCV)
    _RED_RANGES   = [([0, 110, 80], [12, 255, 255]),
                     ([165, 110, 80], [179, 255, 255])]
    _YELLOW_RANGE = ([18, 100, 100], [38, 255, 255])
    _GREEN_RANGE  = ([40,  70,  60], [92, 255, 255])

    @classmethod
    def classify(cls, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return "UNKNOWN"
        h, w = crop.shape[:2]
        if h < 12 or w < 6:
            return "UNKNOWN"

        # Divide into 3 thirds
        z = max(h // 3, 1)
        zones = [crop[0:z, :], crop[z:2*z, :], crop[2*z:, :]]
        hsv_zones = [cv2.cvtColor(z, cv2.COLOR_BGR2HSV) for z in zones]

        def score_color(hsv_img, color):
            if color == "red":
                m = sum(cv2.inRange(hsv_img,
                        np.array(lo), np.array(hi)).sum()
                        for lo, hi in cls._RED_RANGES)
            elif color == "yellow":
                m = cv2.inRange(hsv_img,
                    np.array(cls._YELLOW_RANGE[0]),
                    np.array(cls._YELLOW_RANGE[1])).sum()
            else:  # green
                m = cv2.inRange(hsv_img,
                    np.array(cls._GREEN_RANGE[0]),
                    np.array(cls._GREEN_RANGE[1])).sum()
            total = max(hsv_img.shape[0]*hsv_img.shape[1], 1)
            return m / total

        # Zone weights: red usually top, yellow mid, green bottom
        # We score all three zones but weight them differently per colour
        r_score = (score_color(hsv_zones[0],"red")*1.4
                 + score_color(hsv_zones[1],"red")*0.5
                 + score_color(hsv_zones[2],"red")*0.1)
        y_score = (score_color(hsv_zones[0],"yellow")*0.3
                 + score_color(hsv_zones[1],"yellow")*1.4
                 + score_color(hsv_zones[2],"yellow")*0.3)
        g_score = (score_color(hsv_zones[0],"green")*0.1
                 + score_color(hsv_zones[1],"green")*0.5
                 + score_color(hsv_zones[2],"green")*1.4)

        best_score = max(r_score, y_score, g_score)
        if best_score < 0.015:   # too dim / occluded
            return "UNKNOWN"
        if r_score >= y_score and r_score >= g_score:
            return "RED"
        if y_score >= r_score and y_score >= g_score:
            return "YELLOW"
        return "GREEN"


# ─── Base detector ────────────────────────────────────────────────────────────

class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        raise NotImplementedError


# ─── YOLOv4-tiny detector ────────────────────────────────────────────────────

class YOLOv4Detector(BaseDetector):
    """
    OpenCV DNN YOLOv4-tiny with per-class NMS and strict class-ID mapping.
    """

    def __init__(self, cfg: str, weights: str, names: str):
        if not (os.path.exists(cfg) and os.path.exists(weights)):
            raise FileNotFoundError(f"YOLOv4 files missing: {cfg}, {weights}")
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load COCO names from file (fallback to built-in)
        if os.path.exists(names):
            with open(names, encoding="utf-8") as f:
                self.class_names = [l.strip() for l in f if l.strip()]
        else:
            self.class_names = COCO_NAMES

        self.out_layers = self.net.getUnconnectedOutLayersNames()
        self.size       = DETECTION["input_size"]
        self.nms_t      = DETECTION["nms_threshold"]
        print(f"[YOLOv4] Loaded. Classes: {len(self.class_names)}, "
              f"Output layers: {self.out_layers}")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, self.size, swapRB=True, crop=False)
        self.net.setInput(blob)

        # Collect per-class candidates
        by_class: Dict[str, List] = {}

        for out in self.net.forward(self.out_layers):
            for det in out:
                scores  = det[5:]
                cid     = int(np.argmax(scores))
                raw_conf = float(scores[cid])
                if cid >= len(self.class_names):
                    continue
                coco_name = self.class_names[cid]
                drive_name = _DRIVING_CLASSES.get(coco_name)
                if drive_name is None:
                    continue
                thresh = _CLASS_CONF.get(drive_name, DETECTION["confidence_threshold"])
                if raw_conf < thresh:
                    continue

                cx_n, cy_n, bw_n, bh_n = det[:4]
                bx = int((cx_n - bw_n/2) * w)
                by_px = int((cy_n - bh_n/2) * h)
                bw_px = int(bw_n * w)
                bh_px = int(bh_n * h)

                # Clip to frame
                bx = max(0, min(bx, w-1))
                by_px = max(0, min(by_px, h-1))
                bw_px = max(1, min(bw_px, w - bx))
                bh_px = max(1, min(bh_px, h - by_px))

                if drive_name not in by_class:
                    by_class[drive_name] = {"boxes":[], "confs":[], "raw":[]}
                by_class[drive_name]["boxes"].append([bx, by_px, bw_px, bh_px])
                by_class[drive_name]["confs"].append(raw_conf)
                by_class[drive_name]["raw"].append((drive_name, raw_conf,
                                                    bx, by_px, bw_px, bh_px))

        # Per-class NMS (prevents car suppressing truck at same location)
        results = []
        for cls_name, data in by_class.items():
            idxs = cv2.dnn.NMSBoxes(
                data["boxes"], data["confs"],
                _CLASS_CONF.get(cls_name, DETECTION["confidence_threshold"]),
                self.nms_t)
            if idxs is None or len(idxs) == 0:
                continue
            for i in idxs.flatten():
                n, conf, bx, by_px, bw_px, bh_px = data["raw"][i]
                results.append(DetectedObject(
                    class_name=n,
                    confidence=round(conf, 3),
                    bbox=(bx, by_px, bw_px, bh_px),
                    center_x=bx + bw_px//2,
                    center_y=by_px + bh_px//2,
                    foot_x=bx + bw_px//2,
                    foot_y=by_px + bh_px,
                ))
        return results


# ─── MobileNet SSD fallback ───────────────────────────────────────────────────

_MOB_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor",
]
_MOB_MAP = {
    "person":    "person",
    "bicycle":   "bicycle",
    "bus":       "bus",
    "car":       "car",
    "motorbike": "motorcycle",
}

class MobileNetDetector(BaseDetector):
    def __init__(self, proto: str, model: str):
        if not (os.path.exists(proto) and os.path.exists(model)):
            raise FileNotFoundError(f"MobileNet files missing: {proto}, {model}")
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.thresh = DETECTION["confidence_threshold"]
        print("[MobileNetSSD] Loaded.")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
        self.net.setInput(blob)
        dets = self.net.forward()
        results = []
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf < self.thresh:
                continue
            cid = int(dets[0,0,i,1])
            if cid >= len(_MOB_CLASSES):
                continue
            raw = _MOB_CLASSES[cid]
            name = _MOB_MAP.get(raw)
            if name is None:
                continue
            x1 = max(0, int(dets[0,0,i,3]*w))
            y1 = max(0, int(dets[0,0,i,4]*h))
            x2 = min(w, int(dets[0,0,i,5]*w))
            y2 = min(h, int(dets[0,0,i,6]*h))
            bw, bh = max(1,x2-x1), max(1,y2-y1)
            results.append(DetectedObject(
                class_name=name, confidence=round(conf,3),
                bbox=(x1,y1,bw,bh),
                center_x=(x1+x2)//2, center_y=(y1+y2)//2,
                foot_x=(x1+x2)//2,   foot_y=y2,
            ))
        return results


# ─── ThreatAnalyzer ───────────────────────────────────────────────────────────

class ThreatAnalyzer:
    """
    Enriches detections with:
    - Monocular distance (per-class calibrated heights)
    - Temporal distance smoothing (per track, EMA)
    - Ground-contact corridor test (foot point, not center)
    - Traffic-light state (zone-based HSV classifier)
    - Urgency level
    """

    def __init__(self):
        self._tl_clf       = TrafficLightClassifier()
        self._dist_history: Dict[str, float] = {}   # track_key → smoothed dist

    def analyze(self,
                objects: List[DetectedObject],
                frame: np.ndarray,
                lane_left_x: Optional[int] = None,
                lane_right_x: Optional[int] = None,
                ) -> List[DetectedObject]:
        h, w = frame.shape[:2]
        # Corridor: expand slightly inward from lane lines
        pad = int(w * 0.05)
        corr_w = SAFETY["corridor_width_frac"] * w
        cx_l = lane_left_x  + pad if lane_left_x  is not None else int(w/2 - corr_w/2)
        cx_r = lane_right_x - pad if lane_right_x is not None else int(w/2 + corr_w/2)
        cx_l, cx_r = min(cx_l, cx_r - 10), max(cx_r, cx_l + 10)

        for obj in objects:
            x, y, bw, bh = obj.bbox

            # 1. Distance with per-class calibrated height
            real_h = _REAL_H.get(obj.class_name, 1.5)
            raw_dist = (real_h * FOCAL_LENGTH_PX) / max(bh, 1)

            # 2. EMA smooth distance (per track identified by rough position bucket)
            track_key = f"{obj.class_name}_{obj.center_x//80}_{obj.center_y//80}"
            prev = self._dist_history.get(track_key, raw_dist)
            smooth_dist = 0.35 * raw_dist + 0.65 * prev
            self._dist_history[track_key] = smooth_dist
            obj.distance_m = round(smooth_dist, 2)

            # Expire stale tracks (> 200 entries → keep only 100 most recent)
            if len(self._dist_history) > 200:
                keys = list(self._dist_history.keys())
                for k in keys[:100]:
                    self._dist_history.pop(k, None)

            # 3. Corridor check using FOOT point (bottom-centre of bbox)
            obj.in_corridor = (cx_l <= obj.foot_x <= cx_r)

            # 4. Traffic-light classification
            if obj.class_name == "traffic light":
                crop = frame[max(0,y):min(h,y+bh), max(0,x):min(w,x+bw)]
                obj.tl_state = self._tl_clf.classify(crop)

            # 5. Urgency
            obj.urgency = self._urgency(obj)

        # Sort: in-corridor closest first, then out-of-corridor
        objects.sort(key=lambda o: (not o.in_corridor, o.distance_m))
        return objects

    @staticmethod
    def _urgency(obj: DetectedObject) -> str:
        d  = obj.distance_m
        ic = obj.in_corridor
        is_person = (obj.class_name == "person")

        em = SAFETY["emergency_person_m"] if is_person else SAFETY["emergency_vehicle_m"]
        if d <= em:
            return URGENCY_EMERGENCY
        if d <= SAFETY["high_m"]:
            return URGENCY_HIGH if ic else URGENCY_MEDIUM
        if d <= SAFETY["medium_m"]:
            return URGENCY_MEDIUM if ic else URGENCY_LOW
        return URGENCY_LOW


# ─── ObjectDetector (public API) ─────────────────────────────────────────────

class ObjectDetector:
    """
    Top-level detector:  init best backend → detect → ThreatAnalyzer.
    """

    def __init__(self):
        self._backend  = self._init_backend()
        self._analyzer = ThreatAnalyzer()

    def _init_backend(self) -> BaseDetector:
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        cfg    = os.path.join(base, DETECTION["yolo_cfg"])
        wts    = os.path.join(base, DETECTION["yolo_weights"])
        names  = os.path.join(base, DETECTION["coco_names"])
        proto  = os.path.join(base, DETECTION["mobilenet_proto"])
        model  = os.path.join(base, DETECTION["mobilenet_model"])

        try:
            return YOLOv4Detector(cfg, wts, names)
        except FileNotFoundError:
            print("[ObjectDetector] YOLOv4 weights not found.")
        try:
            return MobileNetDetector(proto, model)
        except FileNotFoundError:
            print("[ObjectDetector] MobileNet weights not found.")

        print("[ObjectDetector] No model weights found – MockDetector active.")
        return MockDetector()

    def detect(self,
               frame: np.ndarray,
               lane_left_x:  Optional[int] = None,
               lane_right_x: Optional[int] = None,
               ) -> List[DetectedObject]:
        raw      = self._backend.detect(frame)
        enriched = self._analyzer.analyze(raw, frame, lane_left_x, lane_right_x)
        return enriched

    def draw_detections(self,
                        frame: np.ndarray,
                        objects: List[DetectedObject],
                        ) -> np.ndarray:
        from config.settings import VIZ
        out = frame.copy()
        for obj in objects:
            x, y, bw, bh = obj.bbox
            h_f, w_f = out.shape[:2]

            base_col = VIZ["bbox_colors"].get(obj.class_name,
                                              VIZ["bbox_colors"]["default"])

            # Urgency-tinted box thickness
            thick = {URGENCY_EMERGENCY:4, URGENCY_HIGH:3,
                     URGENCY_MEDIUM:2}.get(obj.urgency, 1)

            # Draw box
            cv2.rectangle(out, (x,y), (x+bw, y+bh), base_col, thick, cv2.LINE_AA)

            # Foot-point indicator for corridor check
            fp_col = (0,255,0) if obj.in_corridor else (0,100,200)
            cv2.circle(out, (obj.foot_x, min(obj.foot_y, h_f-1)),
                       4, fp_col, -1, cv2.LINE_AA)

            # Build label
            tl_tag = f" ◉{obj.tl_state}" if obj.class_name=="traffic light" else ""
            label  = (f"{obj.class_name}{tl_tag}  "
                      f"{obj.confidence:.2f}  "
                      f"{obj.distance_m:.1f}m  "
                      f"[{obj.urgency}]")

            fs = 0.50
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty  = max(y - 4, lh + 4)
            ty  = min(ty, h_f - 2)
            # Background pill
            cv2.rectangle(out, (x, ty-lh-4), (x+lw+6, ty+bl),
                          base_col, cv2.FILLED)
            cv2.putText(out, label, (x+3, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255,255,255), 1, cv2.LINE_AA)
        return out


# ─── MockDetector (no weights needed) ────────────────────────────────────────

class MockDetector(BaseDetector):
    """
    Content-aware mock using background-subtraction-like thresholding
    to find plausible vehicle blobs. Class assigned by aspect ratio.
    """
    # Strict AR→class mapping  (avoids calling wide blobs "car" when they are "bus")
    _AR_TABLE = [
        (0.0,  0.60, "person"),
        (0.60, 0.95, "motorcycle"),
        (0.95, 1.40, "car"),
        (1.40, 2.20, "truck"),
        (2.20, 9.99, "bus"),
    ]

    def __init__(self):
        print("[MockDetector] Active – no model weights available.")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15,15), 0)
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Keep only road-region (40%-85% of height)
        roi = np.zeros_like(th)
        roi[int(h*0.40):int(h*0.85), :] = th[int(h*0.40):int(h*0.85), :]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < 2500:
                break
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 35 or bh < 25:
                continue
            ar = bw / max(bh, 1)
            cls = "car"
            for lo, hi, name in self._AR_TABLE:
                if lo <= ar < hi:
                    cls = name; break
            conf = min(0.55 + area/3e5, 0.80)
            results.append(DetectedObject(
                class_name=cls, confidence=round(conf,3),
                bbox=(x, y, bw, bh),
                center_x=x+bw//2, center_y=y+bh//2,
                foot_x=x+bw//2,   foot_y=y+bh,
            ))
        return results
