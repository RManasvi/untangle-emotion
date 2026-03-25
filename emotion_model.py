"""Emotion model wrapper.

Provides a lightweight EmotionModel class that delegates to
FaceMeshProcessor + FeatureExtractor + StressEstimator for
per-frame emotion / stress classification.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

import os
import sys

# Ensure local imports work regardless of how it's called
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

try:
    from face_mesh_module import FaceMeshProcessor
    from feature_engineering import FeatureExtractor
    from stress_model import StressEstimator, StressScore
except ImportError:
    from .face_mesh_module import FaceMeshProcessor
    from .feature_engineering import FeatureExtractor
    from .stress_model import StressEstimator, StressScore


class EmotionModel:
    """Thin facade: frame ➜ emotion / stress analysis result."""

    def __init__(self) -> None:
        self.processor = FaceMeshProcessor()
        self.extractor = FeatureExtractor()
        self.estimator = StressEstimator()

    def predict(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """Analyse a single BGR frame.

        Returns a dict with emotion/stress fields, or ``None`` when
        no face is detected in *frame_bgr*.
        """
        landmark_frame = self.processor.process(frame_bgr)
        if landmark_frame is None:
            return None

        features = self.extractor.extract(landmark_frame)
        stress: StressScore = self.estimator.predict(features)
        return {
            "features": features,
            "stress_score": stress.score,
            "stress_level": stress.level,
            "stress_label": stress.label,
            "icon": stress.icon,
        }

    def close(self) -> None:
        self.processor.close()