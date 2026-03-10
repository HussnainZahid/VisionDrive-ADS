"""
Control Layer – Vehicle Controller  (v3 – Professional Grade)
=============================================================
Improvements over v2:
  • Full PID steering (proportional + integral + derivative)
    - Integral term prevents steady-state offset error on long curves
    - Anti-windup clamping on integral
  • Feed-forward lane angle correction (anticipates curve before error builds)
  • Speed PID (proportional + integral) with anti-windup
  • Brake controller: proportional + feedforward for smooth stops
  • Confidence-weighted steering: reduces authority when lane detection is poor
  • Hard-stop logic: forces full brake if state=EMERGENCY regardless of PID
"""

import time
import math
from dataclasses import dataclass
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception.lane_detector   import LaneData
from decision.behavior_engine   import DrivingDecision, DriveState
from config.settings            import CONTROL, FRAME_WIDTH


# ─── VehicleCommand ──────────────────────────────────────────────────────────

@dataclass
class VehicleCommand:
    steering:  float = 0.0    # –1 full-left … +1 full-right
    throttle:  float = 0.0    # 0 … 1
    brake:     float = 0.0    # 0 … 1
    speed_kmh: float = 0.0


# ─── VehicleState (integrator) ────────────────────────────────────────────────

class VehicleState:
    """Physics integrator – replaced by real speed from simulator if available."""

    def __init__(self):
        self._spd = 0.0   # km/h

    def update(self, throttle: float, brake: float, dt: float):
        # Rough kinematics: max accel ~4 m/s², max decel ~9 m/s²
        if brake > 0.05:
            a = -9.0 * brake
        elif throttle > 0.01:
            # Quadratic rolloff so we don't overshoot
            frac = max(0.0, 1.0 - self._spd / max(CONTROL["target_speed_kmh"]*1.6, 1))
            a = 4.0 * throttle * frac
        else:
            a = -0.8   # coast friction

        self._spd = max(0.0, self._spd + a * dt * 3.6)

    def set(self, kmh: float):
        self._spd = max(0.0, kmh)

    @property
    def kmh(self): return self._spd


# ─── PID helpers ─────────────────────────────────────────────────────────────

class PID:
    """Generic discrete PID with anti-windup."""

    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0, windup=50.0):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.windup  = windup
        self._int = 0.0
        self._prev_err = 0.0
        self._prev_t   = time.time()

    def step(self, error: float, dt: float = None) -> float:
        now = time.time()
        dt  = dt or max(now - self._prev_t, 1e-3)
        self._prev_t = now

        self._int = max(-self.windup,
                        min(self.windup, self._int + error * dt))
        deriv = (error - self._prev_err) / dt
        self._prev_err = error

        out = self.kp*error + self.ki*self._int + self.kd*deriv
        return max(self.out_min, min(self.out_max, out))

    def reset(self):
        self._int = 0.0; self._prev_err = 0.0


# ─── Steering Controller ─────────────────────────────────────────────────────

class SteeringController:
    """
    PID on lane_center_offset + feed-forward lane_angle.
    Output EMA-smoothed and confidence-weighted.
    """

    def __init__(self):
        self._pid     = PID(
            kp=CONTROL["steering_kp"],
            ki=CONTROL["steering_kp"]*0.05,   # small integral
            kd=CONTROL["steering_kd"],
            out_min=-1.0, out_max=1.0, windup=30.0)
        self._smooth  = CONTROL["steering_smooth"]
        self._prev    = 0.0

    def compute(self, lane: LaneData, decision: DrivingDecision) -> float:
        # Error = lane offset + any bias the planner requests
        error = lane.lane_center_offset + decision.steering_bias

        # Feed-forward: compensate heading angle before error accumulates
        ff = lane.lane_angle * 0.004

        raw = self._pid.step(error) + ff

        # Confidence weighting: if lane detection is unreliable, reduce authority
        conf = max(lane.confidence, 0.15)
        raw *= conf

        raw = max(-1.0, min(1.0, raw))

        # EMA smoothing
        out = self._smooth * raw + (1-self._smooth) * self._prev
        self._prev = out
        return out

    def reset(self):
        self._pid.reset()
        self._prev = 0.0


# ─── Speed Controller ─────────────────────────────────────────────────────────

class SpeedController:
    """
    PID on speed error → throttle.
    Brake is computed from decision.brake_force (planner-requested)
    plus a proportional penalty when we're over-speed.
    """

    def __init__(self):
        self._pid = PID(
            kp=CONTROL["speed_kp"],
            ki=CONTROL["speed_kp"]*0.02,
            kd=0.0,
            out_min=0.0, out_max=CONTROL["max_throttle"],
            windup=20.0)
        self._prev_thr = 0.0
        self._prev_brk = 0.0
        self._sm       = 0.25   # EMA alpha

    def compute(self, current: float,
                decision: DrivingDecision) -> tuple:
        if decision.brake_force > 0.05:
            # Planner explicitly brakes
            throttle = 0.0
            brake    = min(float(decision.brake_force), CONTROL["max_brake"])
        else:
            err      = decision.target_speed - current
            if err > 0:
                throttle = self._pid.step(err) * decision.throttle_frac
                brake    = 0.0
            else:
                self._pid.reset()   # don't wind up integral while braking
                throttle = 0.0
                brake    = min(-err * 0.06, 0.40)   # gentle aero-brake

        # Smooth
        throttle = self._sm*throttle + (1-self._sm)*self._prev_thr
        brake    = self._sm*brake    + (1-self._sm)*self._prev_brk
        self._prev_thr = throttle
        self._prev_brk = brake
        return throttle, brake


# ─── VehicleController ───────────────────────────────────────────────────────

class VehicleController:
    """
    Combines SteeringController + SpeedController.
    Exposes .step(lane, decision, dt) → VehicleCommand.
    """

    def __init__(self):
        self._steer = SteeringController()
        self._speed = SpeedController()
        self._state = VehicleState()
        self._prev_t = time.time()

    def step(self, lane: LaneData,
             decision: DrivingDecision,
             dt: float = 0.05) -> VehicleCommand:

        # Hard-stop override: emergency or explicit full-brake
        if decision.state in (DriveState.EMERGENCY_BRAKE,
                               DriveState.EMERGENCY_CLEAR):
            self._steer.reset()
            self._state.update(0.0, 1.0, dt)
            return VehicleCommand(
                steering=0.0, throttle=0.0,
                brake=1.0, speed_kmh=self._state.kmh)

        steer          = self._steer.compute(lane, decision)
        throttle, brake = self._speed.compute(self._state.kmh, decision)
        self._state.update(throttle, brake, dt)

        return VehicleCommand(
            steering  = round(steer,   4),
            throttle  = round(throttle,4),
            brake     = round(brake,   4),
            speed_kmh = round(self._state.kmh, 2),
        )

    def inject_speed(self, kmh: float):
        self._state.set(kmh)

    @property
    def current_speed_kmh(self) -> float:
        return self._state.kmh
