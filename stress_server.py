print("DEBUG __name__ =", __name__)
import os
import sys
import time
import base64
import cv2  # type: ignore
import numpy as np  # type: ignore
import threading
from flask import Flask, request, jsonify  # type: ignore
from collections import deque
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

try:
    from face_mesh_module import FaceMeshProcessor, LandmarkFrame  # type: ignore
    from feature_engineering import FeatureExtractor  # type: ignore
    from stress_model import StressEstimator  # type: ignore
except ImportError:
    from .face_mesh_module import FaceMeshProcessor, LandmarkFrame  # type: ignore
    from .feature_engineering import FeatureExtractor  # type: ignore
    from .stress_model import StressEstimator  # type: ignore

app = Flask(__name__)

# ── Global model instances ───────────────────────────────────────────────────
# models_lock protects write access to the global references.
# We initialise lazily but call init_models() eagerly in __main__,
# so the lock is only ever contended on the very first request if the
# server somehow starts without __main__ (e.g. gunicorn).
models_lock = threading.Lock()
processor: Optional[FaceMeshProcessor] = None
extractor: Optional[FeatureExtractor] = None
estimator: Optional[StressEstimator] = None
models_ready = False

# ── History buffers ──────────────────────────────────────────────────────────
stress_history: deque = deque(maxlen=120)
voice_history: deque = deque(maxlen=120)

# ── Shared latest result (returned by GET /analyze) ──────────────────────────
# "stress_label" = "Not Started" signals that no session has run yet.
last_result = {
    "timestamp": 0,
    "eyebrow_raise": 0,
    "lip_tension": 0,
    "head_nod_intensity": 0,
    "symmetry_delta": 0,
    "blink_rate": 0,
    "stress_score": 0,
    "stress_level": "idle",
    "stress_label": "Not Started",   # Sentinel – frontend checks this
    "icon": "🟢",
    "voice_stress": 0,
    "is_mic_active": False,
    "voice_history": [],
    "stress_history": [],
    "is_active": False,              # <-- new: explicit activity flag
}


def _ensure_models():
    """Initialise models if not already done (thread-safe, non-deadlocking)."""
    global processor, extractor, estimator, models_ready
    if models_ready:
        return True
    with models_lock:
        if models_ready:   # Double-checked locking
            return True
        try:
            init_models()
            return True
        except Exception as e:
            print(f"[stress_server] Model init failed: {e}")
            return False


@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response


@app.route("/")
def index():
    return jsonify({
        "service": "AI Micro-Expression Analyzer",
        "version": "2.1.0",
        "status": "online"
    })


@app.route("/health")
def health_check():
    return jsonify({
        "status": "running",
        "models_loaded": models_ready
    })


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_frame():
    global last_result, stress_history, voice_history

    if request.method == "OPTIONS":
        return jsonify({}), 200

    # Ensure models are ready (lazy fallback if __main__ wasn't used)
    if not _ensure_models():
        return jsonify({"error": "Models failed to initialize"}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        img_b64: str = data.get("image") or data.get("frame") or ""
        voice_val: float = float(data.get("voice_stress", 0) or 0)
        is_mic_active: bool = bool(data.get("is_mic_active", False))

        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        # Strip data-URL prefix if present
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        frame_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Mirror and resize
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        # Process landmarks – read under lock to avoid race on processor
        with models_lock:
            if processor is None:
                return jsonify({"error": "Processor not ready"}), 500
            landmark_frame = processor.process(frame)

        if landmark_frame is None:
            # Face not detected – return last payload so report page stays alive
            return jsonify({"error": "No face detected"}), 200

        # These are never None if _ensure_models() returned True
        assert extractor is not None
        assert estimator is not None

        features = extractor.extract(landmark_frame)
        stress_result = estimator.predict(features)

        # Update history
        stress_history.append(stress_result.score)
        voice_history.append(voice_val)

        response_body = {
            "timestamp": int(landmark_frame.timestamp * 1000),
            "eyebrow_raise": round(float(features["eyebrow_raise"]), 4),  # type: ignore
            "lip_tension": round(float(features["lip_tension"]), 4),  # type: ignore
            "head_nod_intensity": round(float(features["head_nod_intensity"]), 4),  # type: ignore
            "symmetry_delta": round(float(features["symmetry_delta"]), 4),  # type: ignore
            "blink_rate": round(float(features["blink_rate"]), 2),  # type: ignore
            "stress_score": round(float(stress_result.score), 3),  # type: ignore
            "stress_level": stress_result.level,
            "stress_label": stress_result.label,
            "icon": stress_result.icon,
            "stress_history": list(stress_history),
            "landmarks": landmark_frame.landmarks.tolist(),
            "voice_stress": round(voice_val, 3),  # type: ignore
            "is_mic_active": is_mic_active,
            "voice_history": list(voice_history),
            "is_active": True,   # Live session flag
        }

        last_result = response_body
        return jsonify(response_body), 200

    except Exception as e:
        print(f"[stress_server] Error in analyze_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["GET"])
def get_analyze():
    """Returns the most recent analysis result (polled by the report page)."""
    return jsonify(last_result)


def init_models():
    global processor, extractor, estimator, models_ready
    print("[stress_server] Initializing models...")
    processor = FaceMeshProcessor()
    extractor = FeatureExtractor()
    estimator = StressEstimator()
    models_ready = True
    print("[stress_server] Models initialized successfully.")

init_models()

if __name__ == "__main__":
    init_models()
    port = 8000
    print(f"[stress_server] Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)