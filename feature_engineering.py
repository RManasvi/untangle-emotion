from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np

import os
import sys

# Ensure local imports work regardless of how it's called
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

try:
    from face_mesh_module import LandmarkFrame
except ImportError:
    from .face_mesh_module import LandmarkFrame

# Landmark indices approximated from MediaPipe Face Mesh topology
LEFT_EYE_LIDS = (159, 145)
RIGHT_EYE_LIDS = (386, 374)
LEFT_EYE_HORIZONTAL = (33, 133)
RIGHT_EYE_HORIZONTAL = (362, 263)
LEFT_EYEBROW = (55, 107, 46)
RIGHT_EYEBROW = (285, 336, 276)
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
TOP_LIP = 13
BOTTOM_LIP = 14
NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _average_points(indices: List[int], landmarks: np.ndarray) -> np.ndarray:
    points = np.array([landmarks[idx] for idx in indices], dtype=np.float32)
    return points.mean(axis=0)


@dataclass
class TemporalMetric:
    window_seconds: float
    timestamps: Deque[float] = field(default_factory=deque)

    def add(self, timestamp: float) -> None:
        self.timestamps.append(timestamp)
        while self.timestamps and (timestamp - self.timestamps[0]) > self.window_seconds:
            self.timestamps.popleft()

    @property
    def count(self) -> int:
        return len(self.timestamps)


class FeatureExtractor:
    def __init__(
        self,
        smoothing_window: int = 5,
        blink_threshold: float = 0.23,
        blink_window_seconds: float = 60.0,
    ) -> None:
        self.smoothing_window = smoothing_window
        self.blink_threshold = blink_threshold
        self.previous_blink_state = False
        self.metrics_history: Dict[str, Deque[float]] = {
            "eyebrow": deque(maxlen=smoothing_window),
            "lip_tension": deque(maxlen=smoothing_window),
            "nod": deque(maxlen=smoothing_window),
            "symmetry": deque(maxlen=smoothing_window),
        }
        self.blink_events = TemporalMetric(window_seconds=blink_window_seconds)
        self.previous_nose_height: float | None = None

    def _eye_aspect_ratio(
        self,
        landmarks: np.ndarray,
        lids: tuple[int, int],
        horizontal_pair: tuple[int, int],
    ) -> float:
        upper = landmarks[lids[0]]
        lower = landmarks[lids[1]]
        horizontal = _distance(landmarks[horizontal_pair[0]], landmarks[horizontal_pair[1]])
        return _distance(upper, lower) / max(horizontal, 1e-5)

    def _compute_blink_rate(self, frame: LandmarkFrame) -> float:
        left_ratio = self._eye_aspect_ratio(
            frame.landmarks,
            LEFT_EYE_LIDS,
            LEFT_EYE_HORIZONTAL,
        )
        right_ratio = self._eye_aspect_ratio(
            frame.landmarks,
            RIGHT_EYE_LIDS,
            RIGHT_EYE_HORIZONTAL,
        )
        eye_ratio = (left_ratio + right_ratio) * 0.5
        is_blinking = eye_ratio < self.blink_threshold
        if is_blinking and not self.previous_blink_state:
            self.blink_events.add(frame.timestamp)
        self.previous_blink_state = is_blinking
        minutes = max(self.blink_events.window_seconds / 60.0, 1e-3)
        return self.blink_events.count / minutes

    def _compute_eyebrow_raise(self, landmarks: np.ndarray) -> float:
        left_brow = _average_points(list(LEFT_EYEBROW), landmarks)
        right_brow = _average_points(list(RIGHT_EYEBROW), landmarks)
        anchor = (landmarks[LEFT_EYE_LIDS[0]] + landmarks[RIGHT_EYE_LIDS[0]]) * 0.5
        left_raise = abs(left_brow[1] - anchor[1])
        right_raise = abs(right_brow[1] - anchor[1])
        value = (left_raise + right_raise) * 0.5
        self.metrics_history["eyebrow"].append(value)
        return float(np.mean(self.metrics_history["eyebrow"]))

    def _compute_lip_tension(self, landmarks: np.ndarray) -> float:
        mouth_width = _distance(landmarks[LEFT_LIP_CORNER], landmarks[RIGHT_LIP_CORNER])
        mouth_height = _distance(landmarks[TOP_LIP], landmarks[BOTTOM_LIP])
        # Normalize: relaxed open mouth ≈ ratio 2–5, neutral closed ≈ 10–30,
        # clenched/pressed lips ≈ 40+
        raw_ratio = mouth_width / max(mouth_height, 1e-5)
        # Map into 0-1: ratio 5 → 0, ratio 60 → 1
        tension = float(np.clip((raw_ratio - 5.0) / 55.0, 0.0, 1.0))
        self.metrics_history["lip_tension"].append(tension)
        return float(np.mean(self.metrics_history["lip_tension"]))

    def _compute_head_nod(self, frame: LandmarkFrame) -> float:
        nose_y = frame.landmarks[NOSE_TIP][1]
        chin_y = frame.landmarks[CHIN][1]
        head_length = abs(chin_y - nose_y)
        if self.previous_nose_height is None:
            self.previous_nose_height = nose_y
            return 0.0
        delta = abs(nose_y - self.previous_nose_height) / max(head_length, 1e-5)
        self.previous_nose_height = nose_y
        self.metrics_history["nod"].append(delta)
        return float(np.mean(self.metrics_history["nod"]))

    def _compute_symmetry(self, landmarks: np.ndarray) -> float:
        left_cheek = landmarks[LEFT_CHEEK]
        right_cheek = landmarks[RIGHT_CHEEK]
        nose = landmarks[NOSE_TIP]
        left_dist = _distance(left_cheek, nose)
        right_dist = _distance(right_cheek, nose)
        symmetry_score = abs(left_dist - right_dist) / max((left_dist + right_dist) * 0.5, 1e-5)
        self.metrics_history["symmetry"].append(symmetry_score)
        return float(np.mean(self.metrics_history["symmetry"]))

    def extract(self, frame: LandmarkFrame) -> Dict[str, float]:
        eyebrow = self._compute_eyebrow_raise(frame.landmarks)
        lip_tension = self._compute_lip_tension(frame.landmarks)
        nod = self._compute_head_nod(frame)
        symmetry = self._compute_symmetry(frame.landmarks)
        blink_rate = self._compute_blink_rate(frame)
        return {
            "eyebrow_raise": eyebrow,
            "lip_tension": lip_tension,
            "head_nod_intensity": nod,
            "symmetry_delta": symmetry,
            "blink_rate": blink_rate,
        }
