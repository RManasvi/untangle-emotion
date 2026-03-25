from __future__ import annotations
import os
import sys
import argparse
import pathlib
from typing import Dict, Tuple

import cv2
import numpy as np

# Ensure path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from emotion_model import EmotionModel
    import data_logger
    from dashboard import Dashboard
    from face_mesh_module import iter_landmarks_from_camera, LandmarkFrame
    from feature_engineering import FeatureExtractor
    from stress_model import StressEstimator, StressScore
except ImportError:
    # Fallback to local
    from .emotion_model import EmotionModel
    from . import data_logger
    from .dashboard import Dashboard
    from .face_mesh_module import iter_landmarks_from_camera, LandmarkFrame
    from .feature_engineering import FeatureExtractor
    from .stress_model import StressEstimator, StressScore

# ── Colour palette (BGR) ────────────────────────────────────────────
COLORS: Dict[str, Tuple[int, int, int]] = {
    "calm": (0, 200, 0),       # green
    "mild": (0, 200, 255),     # amber / yellow
    "high": (0, 0, 230),       # red
}
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (60, 60, 60)
DARK_BG = (30, 30, 30)
BAR_BG = (50, 50, 50)
LANDMARK_COLOR = (200, 220, 200)

WINDOW = "AI Micro-Expression Analyzer"
PANEL_W = 320  # width of side panel


# ── Draw face-mesh dots on the camera image ─────────────────────────
def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> None:
    h, w = image.shape[:2]
    for lm in landmarks:
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image, (x, y), 1, LANDMARK_COLOR, -1)


# ── Draw a horizontal progress bar ──────────────────────────────────
def draw_bar(
    panel: np.ndarray,
    x: int,
    y: int,
    bar_w: int,
    bar_h: int,
    ratio: float,
    color: Tuple[int, int, int],
) -> None:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(panel, (x, y), (x + bar_w, y + bar_h), BAR_BG, -1)
    fill_w = int(bar_w * ratio)
    if fill_w > 0:
        cv2.rectangle(panel, (x, y), (x + fill_w, y + bar_h), color, -1)
    cv2.rectangle(panel, (x, y), (x + bar_w, y + bar_h), WHITE, 1)


# ── Build the side panel ────────────────────────────────────────────
def build_panel(
    height: int,
    features: Dict[str, float],
    stress: StressScore,
    emotion: str,   # 👈 add this
) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), DARK_BG, dtype=np.uint8)
    color = COLORS.get(stress.level, WHITE)

    # ── Stress banner ───────────────────────────────────────────
    banner_h = 110
    cv2.rectangle(panel, (0, 0), (PANEL_W, banner_h), color, -1)

    level_text = stress.label.upper()
    cv2.putText(panel, level_text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)

    cv2.putText(panel, f"Score: {stress.score:.2f}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1, cv2.LINE_AA)

    cv2.putText(panel, f"Emotion: {emotion}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    # ── Metric bars ─────────────────────────────────────────────
    pretty = {
        "eyebrow_raise": ("Eyebrow Raise", 0.08),
        "lip_tension": ("Lip Tension", 1.0),
        "head_nod_intensity": ("Head Nod", 1.5),
        "symmetry_delta": ("Symmetry", 0.05),
        "blink_rate": ("Blink Rate /min", 30.0),
    }
    y = banner_h + 25
    bar_w = PANEL_W - 40
    bar_h = 18
    for key, (label, scale) in pretty.items():
        val = features.get(key, 0.0)
        ratio = val / scale
        cv2.putText(panel, f"{label}: {val:.3f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        draw_bar(panel, 15, y + 5, bar_w, bar_h, ratio, color)
        y += bar_h + 30

    # ── Legend ──────────────────────────────────────────────────
    y += 10
    for lvl, (r, g, b) in COLORS.items():
        cv2.circle(panel, (25, y), 8, (r, g, b), -1)
        cv2.putText(panel, lvl.capitalize(), (42, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 28

    # ── Instructions ───────────────────────────────────────────
    cv2.putText(panel, "Press 'q' to quit", (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
    return panel


# ── Combine camera feed + panel into one window ─────────────────────
def render_frame(
    frame: LandmarkFrame,
    features: Dict[str, float],
    stress: StressScore,
    emotion: str,
) -> np.ndarray:
    image = frame.image.copy()
    draw_landmarks(image, frame.landmarks)

    border_color = COLORS.get(stress.level, WHITE)
    cv2.rectangle(image, (0, 0),
                  (image.shape[1] - 1, image.shape[0] - 1), border_color, 3)

    panel = build_panel(image.shape[0], features, stress, emotion)
    combined = np.hstack([image, panel])
    return combined


# ── Main loop ───────────────────────────────────────────────────────
def run(camera_index: int, log_path: pathlib.Path, display: bool, verbose: bool) -> None:
    extractor = FeatureExtractor()
    estimator = StressEstimator()
    emotion_model = EmotionModel()
    fields = [
        "eyebrow_raise",
        "lip_tension",
        "head_nod_intensity",
        "symmetry_delta",
        "blink_rate",
        "stress_score",
    ]
    dashboard = Dashboard(verbose=verbose)

    with data_logger.DataLogger(log_path, fieldnames=fields) as logger:
        for frame in iter_landmarks_from_camera(camera_index):
            features = extractor.extract(frame)
            emotion = emotion_model.predict(frame.image)
            print("DEEPFACE EMOTION:", emotion)
            stress_score = estimator.predict(features)
            metrics = {**features, "stress_score": stress_score.score}

            # Terminal output (always)
            dashboard.render(features, stress_score)
            logger.log(metrics)

            # OpenCV visual output
            if display and frame.image is not None:
                canvas = render_frame(frame, features, stress_score,emotion)
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time micro-expression stress analyzer"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--log-path",
        type=pathlib.Path,
        default=pathlib.Path("logs/session.csv"),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview window",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full metric breakdown to terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        camera_index=args.camera_index,
        log_path=args.log_path,
        display=not args.no_display,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
