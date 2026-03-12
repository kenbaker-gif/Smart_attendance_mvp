import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import time

# -----------------------------
# Path Configuration
# -----------------------------
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent.parent
DATA_DIR = APP_DIR / "streamlit" / "data"
ENCODINGS_PATH = DATA_DIR / "encodings/encodings_insightface.pkl"
DEFAULT_THRESHOLD = 0.5
LIVENESS_THRESHOLD = 0.6  # Above this = live, below = spoof

# -----------------------------
# GLOBAL MEMORY (RAM)
# -----------------------------
_app = None
_antispoof = None
_CACHE_ENCODINGS = np.array([])
_CACHE_IDS = []


def get_insightface(det_size=(320, 320), model_name="buffalo_s"):
    """
    ✅ Speed optimization: det_size reduced from (640,640) to (320,320)
    Cuts detection time ~50% with minimal accuracy loss on mobile photos.
    """
    global _app
    if _app is not None:
        return _app

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("Please run: pip install insightface onnxruntime")

    _app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=det_size)
    print("✅ FaceAnalysis model loaded (det_size=320x320).")
    return _app


def get_antispoof():
    """Load MiniFASNetV2 anti-spoofing model via uniface."""
    global _antispoof
    if _antispoof is not None:
        return _antispoof

    try:
        from uniface import create_spoofer
        _antispoof = create_spoofer()  # downloads MiniFASNetV2 automatically
        print("✅ Anti-spoof model loaded.")
    except Exception as e:
        print(f"⚠️ Anti-spoof model failed to load: {e}. Liveness check disabled.")
        _antispoof = None

    return _antispoof


def preload_models():
    """
    ✅ Speed optimization: preload both models at server startup
    so first scan is as fast as subsequent scans.
    """
    print("🔄 Preloading face analysis model...")
    get_insightface()
    print("🔄 Preloading anti-spoof model...")
    get_antispoof()
    print("✅ All models preloaded and ready.")


def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def update_face_bank(new_data_dict: dict):
    """
    Updates the in-memory cache with new data from Supabase.
    Args:
        new_data_dict: { 'student_id': [0.123, 0.456, ...] }
    """
    global _CACHE_ENCODINGS, _CACHE_IDS

    if not new_data_dict:
        print("⚠️ Engine: Received empty data update.")
        return

    print(f"🧠 Engine: Updating memory with {len(new_data_dict)} faces...")

    try:
        ids = list(new_data_dict.keys())
        embeddings = list(new_data_dict.values())
        emb_array = np.array(embeddings, dtype=np.float32)
        _CACHE_ENCODINGS = normalize_encodings(emb_array)
        _CACHE_IDS = ids
        print(f"✅ Engine: Memory Updated! Holding {len(_CACHE_IDS)} students.")
    except Exception as e:
        print(f"❌ Engine Update Error: {e}")


def get_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract face embedding from an image — used during registration."""
    app = get_insightface()
    faces = app.get(img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = face.embedding
    if embedding is None:
        return None
    return embedding / (np.linalg.norm(embedding) + 1e-10)


def check_liveness(img_bgr: np.ndarray, face) -> Tuple[bool, float]:
    """
    Run anti-spoofing check on a detected face using uniface MiniFASNetV2.
    Returns (is_live, score) where score > LIVENESS_THRESHOLD = live person.
    Falls back to True if model not available.
    """
    antispoof = get_antispoof()

    if antispoof is None:
        return True, 1.0

    try:
        bbox = face.bbox.astype(int).tolist()
        result = antispoof.predict(img_bgr, bbox)
        is_live = result.is_real and result.confidence > LIVENESS_THRESHOLD
        return is_live, float(result.confidence)

    except Exception as e:
        print(f"⚠️ Liveness check error: {e}. Defaulting to live.")
        return True, 1.0

def verify_face(img_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Optional[dict]:
    global _CACHE_ENCODINGS, _CACHE_IDS

    if _CACHE_ENCODINGS.size == 0:
        return {"status": "error", "message": "Server is warming up... Try again in 10s."}

    app = get_insightface()

    t1 = time.time()
    faces = app.get(img_bgr)
    t2 = time.time()

    if not faces:
        return None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    t3 = time.time()
    is_live, liveness_score = check_liveness(img_bgr, face)
    t4 = time.time()

    print(f"⏱️ Detection: {t2-t1:.2f}s | Liveness: {t4-t3:.2f}s | Total: {t4-t1:.2f}s")
    # ... rest of function unchanged

    # 4. ✅ Liveness check — reject photos/screens
    is_live, liveness_score = check_liveness(img_bgr, face)

    bbox = face.bbox.astype(int).tolist()
    kps  = face.kps.astype(int).tolist()

    if not is_live:
        print(f"🚫 Spoof detected! Liveness score: {liveness_score:.2f}")
        return {
            "status":         "spoof",
            "message":        "Spoof detected. Please use your real face.",
            "liveness_score": round(liveness_score, 2),
            "confidence":     0.0,
            "bbox":           bbox,
            "kps":            kps,
        }

    # 5. Get embedding
    captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)

    # 6. Compare with memory (vectorized cosine similarity)
    dists = 1.0 - np.dot(_CACHE_ENCODINGS, captured_emb)
    idx   = np.argmin(dists)
    score = float(1.0 - dists[idx])

    if dists[idx] < threshold:
        return {
            "status":         "success",
            "student_id":     _CACHE_IDS[idx],
            "confidence":     score,
            "liveness_score": round(liveness_score, 2),
            "bbox":           bbox,
            "kps":            kps,
        }
    else:
        return {
            "status":         "failed",
            "student_id":     "Unknown",
            "confidence":     score,
            "liveness_score": round(liveness_score, 2),
            "bbox":           bbox,
            "kps":            kps,
        }