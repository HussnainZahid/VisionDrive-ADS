"""
Utils – Video I/O
==================
Thin wrappers around OpenCV VideoCapture / VideoWriter.
"""

import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import RECORDING, INPUT, FRAME_WIDTH, FRAME_HEIGHT


class VideoSource:
    """
    Wraps cv2.VideoCapture with looping and resize support.
    Accepts a file path or integer (webcam index).
    """

    def __init__(self, source=None, loop=True):
        source = source if source is not None else INPUT["source"]
        self._loop   = loop
        self._source = source
        self._cap    = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VideoSource] Opened: {source} "
              f"({self.width}×{self.height} @ {self.fps:.1f}fps, "
              f"{self.total_frames} frames)")

    def read(self):
        """Return next BGR frame, or None at end (if not looping)."""
        ok, frame = self._cap.read()
        if not ok:
            if self._loop and self.total_frames > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        return frame

    def release(self):
        self._cap.release()

    @property
    def position(self):
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))


class VideoRecorder:
    """
    Wraps cv2.VideoWriter.
    Call .write(frame) every iteration; .release() when done.
    """

    def __init__(self, path=None, fps=None, width=None, height=None):
        path   = path   or RECORDING["output_path"]
        fps    = fps    or RECORDING["fps"]
        width  = width  or FRAME_WIDTH
        height = height or FRAME_HEIGHT

        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*RECORDING["codec"])
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        self._count  = 0
        self._max    = int(fps * RECORDING["duration_sec"])
        self._path   = path
        print(f"[VideoRecorder] Recording to: {path} "
              f"({width}×{height} @ {fps}fps, max {self._max} frames)")

    def write(self, frame):
        if self._count >= self._max:
            return False   # signal: stop recording
        self._writer.write(frame)
        self._count += 1
        return True

    def release(self):
        self._writer.release()
        print(f"[VideoRecorder] Saved {self._count} frames → {self._path}")

    @property
    def is_full(self):
        return self._count >= self._max

    @property
    def frames_written(self):
        return self._count
