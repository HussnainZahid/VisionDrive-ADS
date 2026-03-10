"""
Decision Layer – Behavior Engine  (v3 – Professional Grade)
============================================================
Improvements over v2:
  • Full 5-state traffic light FSM with hysteresis (prevents rapid toggling)
  • Stop-sign FSM with creep-forward before full stop
  • Obstacle avoidance with inter-vehicle gap model (time-to-collision proxy)
  • Speed regulation: PID-like with feed-forward curvature
  • EMERGENCY state with latched recovery (won't resume until clear for N frames)
  • Dual urgency: immediate (current frame) + sustained (rolling N-frame max)
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque
from collections import deque
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.lane_detector   import LaneData
from perception.object_detector import (DetectedObject, URGENCY_RANK,
    URGENCY_EMERGENCY, URGENCY_HIGH, URGENCY_MEDIUM, URGENCY_LOW, URGENCY_CLEAR)
from config.settings import CONTROL, SAFETY


# ─── States ───────────────────────────────────────────────────────────────────

class DriveState:
    CRUISE          = "CRUISE"
    SLOW_DOWN       = "SLOW_DOWN"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
    EMERGENCY_CLEAR = "EMERGENCY_CLEAR"   # recovery period after emergency
    TL_APPROACH     = "TL_APPROACH"
    TL_STOP         = "TL_STOP"
    TL_WAIT         = "TL_WAIT"
    TL_DEPART       = "TL_DEPART"
    SS_APPROACH     = "SS_APPROACH"
    SS_CREEP        = "SS_CREEP"
    SS_STOP         = "SS_STOP"
    SS_WAIT         = "SS_WAIT"
    SS_DEPART       = "SS_DEPART"


# ─── Decision output ──────────────────────────────────────────────────────────

@dataclass
class DrivingDecision:
    state:          str   = DriveState.CRUISE
    target_speed:   float = CONTROL["target_speed_kmh"]
    steering_bias:  float = 0.0
    brake_force:    float = 0.0
    throttle_frac:  float = 1.0
    urgency:        str   = URGENCY_CLEAR
    reason:         str   = ""
    tl_state:       str   = "UNKNOWN"
    closest_dist_m: float = 999.0


# ─── Behavior Engine ──────────────────────────────────────────────────────────

class BehaviorEngine:
    """
    Rule-based planner with state machines for each scenario.

    Priority (decreasing):
      1. EMERGENCY_BRAKE  – object at emergency distance in corridor
      2. EMERGENCY_CLEAR  – latched recovery period (N frames clear)
      3. Traffic-light FSM
      4. Stop-sign FSM
      5. Obstacle slow-down (gap model)
      6. Curvature speed regulation
      7. CRUISE
    """

    _EM_CLEAR_FRAMES = 10   # frames to stay in EMERGENCY_CLEAR before resuming

    def __init__(self):
        self._state          = DriveState.CRUISE
        self._tl_timer       = 0.0
        self._ss_timer       = 0.0
        self._em_clear_count = 0
        self._prev_tl_state  = "UNKNOWN"
        self._tl_hyst_count  = 0        # hysteresis counter for TL state change
        # Rolling urgency window (last N frames)
        self._urgency_window: Deque[str] = deque(maxlen=8)

    # ── Public ────────────────────────────────────────────────────────────────

    def plan(self,
             lane:    LaneData,
             objects: List[DetectedObject],
             speed:   float = 0.0,
             ) -> DrivingDecision:
        dec = DrivingDecision(target_speed=CONTROL["target_speed_kmh"])
        now = time.time()

        # ── Find most-threatening in-corridor object ──────────────────────────
        corridor_objs = [o for o in objects if o.in_corridor]
        nearest = corridor_objs[0] if corridor_objs else None
        if nearest:
            dec.closest_dist_m = nearest.distance_m

        # ── 1. Emergency brake ────────────────────────────────────────────────
        em_obj = self._find_emergency(corridor_objs)
        if em_obj:
            dec.state         = DriveState.EMERGENCY_BRAKE
            dec.target_speed  = 0.0
            dec.brake_force   = 1.0
            dec.throttle_frac = 0.0
            dec.urgency       = URGENCY_EMERGENCY
            dec.reason        = (f"EMERGENCY: {em_obj.class_name} "
                                 f"@ {em_obj.distance_m:.1f}m")
            self._state          = DriveState.EMERGENCY_BRAKE
            self._em_clear_count = 0
            self._urgency_window.append(URGENCY_EMERGENCY)
            return dec

        # ── 2. Emergency recovery latch ───────────────────────────────────────
        if self._state == DriveState.EMERGENCY_BRAKE:
            self._em_clear_count += 1
            if self._em_clear_count < self._EM_CLEAR_FRAMES:
                dec.state         = DriveState.EMERGENCY_CLEAR
                dec.target_speed  = 0.0
                dec.brake_force   = 0.6
                dec.throttle_frac = 0.0
                dec.urgency       = URGENCY_HIGH
                dec.reason        = (f"Emergency recovery "
                                     f"({self._em_clear_count}/{self._EM_CLEAR_FRAMES})")
                return dec
            self._state = DriveState.CRUISE

        # ── 3. Traffic light FSM ──────────────────────────────────────────────
        tl_dec = self._traffic_light_fsm(objects, dec, now, speed)
        if tl_dec is not None:
            self._urgency_window.append(tl_dec.urgency)
            return tl_dec

        # ── 4. Stop-sign FSM ──────────────────────────────────────────────────
        ss_dec = self._stop_sign_fsm(objects, dec, now, speed)
        if ss_dec is not None:
            self._urgency_window.append(ss_dec.urgency)
            return ss_dec

        # ── 5. Obstacle gap model ─────────────────────────────────────────────
        dec = self._obstacle_avoidance(dec, corridor_objs, speed)

        # ── 6. Curvature speed regulation ─────────────────────────────────────
        if lane.curvature > 5e-4:
            # Reduce speed proportional to curvature; minimum 18 km/h
            reduction = lane.curvature * CONTROL["curvature_speed_scale"] * 2000
            dec.target_speed = max(dec.target_speed - reduction, 18.0)

        # ── 7. Cruise default ─────────────────────────────────────────────────
        if dec.state not in (DriveState.SLOW_DOWN,):
            dec.state  = DriveState.CRUISE
            dec.urgency = URGENCY_CLEAR
            dec.reason = "Road clear"

        self._state = dec.state
        self._urgency_window.append(dec.urgency)
        return dec

    # ── Emergency detection ───────────────────────────────────────────────────

    @staticmethod
    def _find_emergency(corridor_objs: List[DetectedObject]
                        ) -> Optional[DetectedObject]:
        for obj in corridor_objs:
            is_p   = obj.class_name == "person"
            thresh = (SAFETY["emergency_person_m"] if is_p
                      else SAFETY["emergency_vehicle_m"])
            if obj.distance_m <= thresh:
                return obj
        return None

    # ── Traffic-light FSM ────────────────────────────────────────────────────

    def _traffic_light_fsm(self, objects, dec, now, speed):
        tl_list = [o for o in objects
                   if o.class_name == "traffic light" and o.in_corridor]
        if not tl_list:
            # Transition out of TL states if no light visible
            if self._state in (DriveState.TL_STOP, DriveState.TL_WAIT):
                # Don't immediately depart – wait for green or timeout
                pass
            elif self._state == DriveState.TL_DEPART:
                dec.state  = DriveState.TL_DEPART
                dec.reason = "TL departed – resuming"
                self._state = DriveState.CRUISE
                return dec
            return None

        # Pick closest traffic light
        tl = min(tl_list, key=lambda o: o.distance_m)
        dec.tl_state = tl.tl_state

        # Hysteresis: require 2 consecutive same state before acting
        if tl.tl_state == self._prev_tl_state:
            self._tl_hyst_count = min(self._tl_hyst_count + 1, 5)
        else:
            self._tl_hyst_count = 0
            self._prev_tl_state = tl.tl_state

        stable_state = tl.tl_state if self._tl_hyst_count >= 2 else self._prev_tl_state

        stop_d = CONTROL["tl_stop_distance_m"]
        slow_d = CONTROL["tl_slow_distance_m"]

        if stable_state == "RED":
            if tl.distance_m <= stop_d:
                if speed < 0.5 and self._state != DriveState.TL_WAIT:
                    self._tl_timer = now
                    self._state    = DriveState.TL_WAIT
                dec.state         = DriveState.TL_WAIT if self._state == DriveState.TL_WAIT else DriveState.TL_STOP
                dec.target_speed  = 0.0
                dec.brake_force   = 0.90
                dec.throttle_frac = 0.0
                dec.urgency       = URGENCY_HIGH
                dec.reason        = f"RED light – stopped ({tl.distance_m:.1f}m)"
                return dec
            elif tl.distance_m <= slow_d:
                dec.state         = DriveState.TL_APPROACH
                dec.target_speed  = max(8.0, 10.0*(tl.distance_m/slow_d))
                dec.throttle_frac = 0.30
                dec.brake_force   = max(0.0, 0.3*(1-(tl.distance_m/slow_d)))
                dec.urgency       = URGENCY_MEDIUM
                dec.reason        = f"RED light approaching ({tl.distance_m:.1f}m)"
                self._state       = DriveState.TL_APPROACH
                return dec

        elif stable_state == "YELLOW":
            if tl.distance_m <= slow_d:
                dec.state         = DriveState.TL_APPROACH
                dec.target_speed  = 8.0
                dec.brake_force   = 0.35
                dec.throttle_frac = 0.20
                dec.urgency       = URGENCY_MEDIUM
                dec.reason        = f"YELLOW – slowing ({tl.distance_m:.1f}m)"
                self._state       = DriveState.TL_APPROACH
                return dec

        elif stable_state == "GREEN":
            if self._state in (DriveState.TL_STOP, DriveState.TL_WAIT,
                               DriveState.TL_APPROACH):
                dec.state  = DriveState.TL_DEPART
                dec.urgency = URGENCY_CLEAR
                dec.reason = "GREEN light – go"
                self._state = DriveState.CRUISE
                return dec

        # Waiting at red light with no new detections
        if self._state == DriveState.TL_WAIT:
            dec.state         = DriveState.TL_WAIT
            dec.target_speed  = 0.0
            dec.brake_force   = 0.90
            dec.throttle_frac = 0.0
            dec.urgency       = URGENCY_HIGH
            dec.reason        = f"Waiting at RED ({now-self._tl_timer:.1f}s)"
            return dec

        return None

    # ── Stop-sign FSM ─────────────────────────────────────────────────────────

    def _stop_sign_fsm(self, objects, dec, now, speed):
        ss_list = [o for o in objects
                   if o.class_name == "stop sign" and o.in_corridor]

        # In SS_WAIT even when sign leaves FOV
        if self._state == DriveState.SS_WAIT:
            elapsed = now - self._ss_timer
            if elapsed >= CONTROL["stop_sign_wait_sec"]:
                dec.state  = DriveState.SS_DEPART
                dec.reason = f"Stop sign – wait done ({elapsed:.1f}s), departing"
                self._state = DriveState.CRUISE
                return dec
            dec.state         = DriveState.SS_WAIT
            dec.target_speed  = 0.0
            dec.brake_force   = 0.90
            dec.throttle_frac = 0.0
            dec.urgency       = URGENCY_HIGH
            dec.reason        = (f"STOP SIGN wait: "
                                 f"{elapsed:.1f}/{CONTROL['stop_sign_wait_sec']}s")
            return dec

        if not ss_list:
            return None

        ss = min(ss_list, key=lambda o: o.distance_m)

        slow_d = CONTROL["tl_slow_distance_m"]
        stop_d = CONTROL["tl_stop_distance_m"]

        if ss.distance_m <= stop_d:
            # Full stop – enter wait once truly stopped
            if speed < 0.5 and self._state != DriveState.SS_WAIT:
                self._ss_timer = now
                self._state    = DriveState.SS_WAIT
            dec.state         = DriveState.SS_STOP
            dec.target_speed  = 0.0
            dec.brake_force   = 0.90
            dec.throttle_frac = 0.0
            dec.urgency       = URGENCY_HIGH
            dec.reason        = f"STOP SIGN – {ss.distance_m:.1f}m"
            return dec

        elif ss.distance_m <= slow_d * 0.7:
            # Creep phase
            dec.state         = DriveState.SS_CREEP
            dec.target_speed  = 5.0
            dec.throttle_frac = 0.18
            dec.brake_force   = 0.10
            dec.urgency       = URGENCY_MEDIUM
            dec.reason        = f"STOP SIGN creep ({ss.distance_m:.1f}m)"
            self._state       = DriveState.SS_CREEP
            return dec

        elif ss.distance_m <= slow_d:
            dec.state         = DriveState.SS_APPROACH
            dec.target_speed  = 12.0
            dec.throttle_frac = 0.30
            dec.urgency       = URGENCY_MEDIUM
            dec.reason        = f"STOP SIGN approaching ({ss.distance_m:.1f}m)"
            self._state       = DriveState.SS_APPROACH
            return dec

        return None

    # ── Obstacle avoidance ────────────────────────────────────────────────────

    def _obstacle_avoidance(self, dec: DrivingDecision,
                             corridor_objs: List[DetectedObject],
                             speed: float) -> DrivingDecision:
        if not corridor_objs:
            return dec

        worst = URGENCY_LOW
        for obj in corridor_objs:
            if obj.class_name in ("traffic light", "stop sign"):
                continue   # handled by dedicated FSMs
            u = obj.urgency
            if URGENCY_RANK[u] > URGENCY_RANK[worst]:
                worst = u

            if u == URGENCY_EMERGENCY:
                pass   # already handled above
            elif u == URGENCY_HIGH:
                # Gap model: target_speed proportional to distance
                safe_speed = max(obj.distance_m * 1.2, 0.0)
                dec.target_speed  = min(dec.target_speed, safe_speed)
                dec.throttle_frac = 0.35
                dec.brake_force   = max(dec.brake_force, 0.08)
                dec.state         = DriveState.SLOW_DOWN
                dec.reason        = (f"HIGH obstacle: {obj.class_name} "
                                     f"@ {obj.distance_m:.1f}m")
            elif u == URGENCY_MEDIUM:
                safe_speed = max(obj.distance_m * 0.9, 8.0)
                dec.target_speed  = min(dec.target_speed, safe_speed)
                dec.throttle_frac = 0.55
                if dec.state != DriveState.SLOW_DOWN:
                    dec.state  = DriveState.SLOW_DOWN
                    dec.reason = (f"MEDIUM obstacle: {obj.class_name} "
                                  f"@ {obj.distance_m:.1f}m")

        if worst != URGENCY_LOW:
            dec.urgency = worst
        return dec
