"""
Utils – Performance Monitor
=============================
Tracks FPS, per-module latency, and system health.
"""

import time
import collections
from typing import Dict


class PerfMonitor:
    """
    Lightweight rolling-window performance tracker.

    Usage:
        mon = PerfMonitor()
        mon.start("lane")
        ... do lane detection ...
        mon.stop("lane")
        stats = mon.stats()
    """

    def __init__(self, window: int = 60):
        self._window  = window
        self._frame_t = collections.deque(maxlen=window)
        self._timers:  Dict[str, float]                       = {}
        self._latency: Dict[str, collections.deque]           = {}
        self._last_frame = time.time()

    def tick(self):
        """Call once per frame to track FPS."""
        now = time.time()
        dt  = now - self._last_frame
        self._last_frame = now
        self._frame_t.append(dt)

    def start(self, name: str):
        self._timers[name] = time.perf_counter()

    def stop(self, name: str):
        if name not in self._timers:
            return
        elapsed = (time.perf_counter() - self._timers[name]) * 1000  # ms
        if name not in self._latency:
            self._latency[name] = collections.deque(maxlen=self._window)
        self._latency[name].append(elapsed)

    def fps(self) -> float:
        if len(self._frame_t) < 2:
            return 0.0
        return 1.0 / (sum(self._frame_t) / len(self._frame_t) + 1e-9)

    def latency_ms(self, name: str) -> float:
        buf = self._latency.get(name)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def stats(self) -> Dict:
        return {
            "fps":    self.fps(),
            "lane_ms":   self.latency_ms("lane"),
            "detect_ms": self.latency_ms("detect"),
            "hud_ms":    self.latency_ms("hud"),
            "total_ms":  self.latency_ms("total"),
        }
