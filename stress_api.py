"""FastAPI endpoint for the AI Micro-Expression Stress Analyzer.

Run with:
    python -m uvicorn ai_microexpression_analyzer.stress.stress_api:app --reload --log-level debug
"""
from __future__ import annotations

import base64
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Relative imports (package context) ─────────────────────────────
try:
    from .face_mesh_module import FaceMeshProcessor
    from .feature_engineering import FeatureExtractor
    from .stress_model import StressEstimator
except ImportError:
    from face_mesh_module import FaceMeshProcessor
    from feature_engineering import FeatureExtractor
    from stress_model import StressEstimator

# ── Global model singletons ────────────────────────────────────────
processor: Optional[FaceMeshProcessor] = None
extractor: Optional[FeatureExtractor] = None
estimator: Optional[StressEstimator] = None

# Stress history buffer (for time-series graph on the frontend)
stress_history: Deque[float] = deque(maxlen=120)

last_result: Dict = {
    "timestamp": 0,
    "eyebrow_raise": 0,
    "lip_tension": 0,
    "head_nod_intensity": 0,
    "symmetry_delta": 0,
    "blink_rate": 0,
    "stress_score": 0,
    "stress_level": "calm",
    "stress_label": "Not Started",
    "icon": "🟢",
}


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy ML models once on startup and clean up on shutdown."""
    global processor, extractor, estimator
    print("🔄 Initialising models …")
    processor = FaceMeshProcessor()
    extractor = FeatureExtractor()
    estimator = StressEstimator()
    print("✅ Models loaded successfully.")
    yield
    # Shutdown: release resources
    if processor is not None:
        processor.close()
    print("🛑 Models released.")


app = FastAPI(
    title="AI Micro-Expression Analyzer",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────
class Base64Frame(BaseModel):
    image: Optional[str] = None
    frame: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "AI Micro-Expression Analyzer",
        "version": "2.0.0",
        "status": "online",
    }


@app.get("/health")
async def health():
    return {"status": "running", "models_loaded": processor is not None}


@app.post("/analyze")
async def analyze(payload: Base64Frame):
    """Accept a base64-encoded image and return stress / micro-expression analysis."""
    global last_result

    img_b64 = payload.image or payload.frame
    if not img_b64:
        return {"error": "No image provided"}, 400

    try:
        # Strip data-URI prefix (e.g. "data:image/jpeg;base64,…")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        frame_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Failed to decode frame"}

        # Mirror & normalise size
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        # Landmark detection
        landmark_frame = processor.process(frame)
        if landmark_frame is None:
            return {"error": "No face detected"}

        # Feature extraction + stress estimation
        features = extractor.extract(landmark_frame)
        stress_result = estimator.predict(features)

        stress_history.append(stress_result.score)

        response = {
            "timestamp": int(landmark_frame.timestamp * 1000),
            "eyebrow_raise": round(features["eyebrow_raise"], 4),
            "lip_tension": round(features["lip_tension"], 4),
            "head_nod_intensity": round(features["head_nod_intensity"], 4),
            "symmetry_delta": round(features["symmetry_delta"], 4),
            "blink_rate": round(features["blink_rate"], 2),
            "stress_score": round(stress_result.score, 3),
            "stress_level": stress_result.level,
            "stress_label": stress_result.label,
            "icon": stress_result.icon,
            "stress_history": list(stress_history),
            "landmarks": landmark_frame.landmarks.tolist(),
        }

        last_result = response
        return response

    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Accept an uploaded image file instead of base64."""
    contents = await file.read()
    b64 = base64.b64encode(contents).decode()
    return await analyze(Base64Frame(image=b64))


@app.get("/analyze")
async def get_last_result():
    """Return the most recent analysis result (polling fallback)."""
    return last_result