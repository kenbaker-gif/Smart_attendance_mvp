import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

# -----------------------------
# Path Configuration
# -----------------------------
CURRENT_FILE = Path(__file__).resolve()
# Go up 2 levels: 'app/face_engine/engine.py' -> 'app/'
APP_DIR = CURRENT_FILE.parent.parent 
# Point to 'app/streamlit/data'
DATA_DIR = APP_DIR / "streamlit" / "data"
ENCODINGS_PATH = DATA_DIR / "encodings/encodings_insightface.pkl"
DEFAULT_THRESHOLD = 0.5

# -----------------------------
# AI Engine
# -----------------------------
_app = None

def get_insightface(det_size=(640, 640), model_name="buffalo_s"):
    global _app
    if _app is not None:
        return _app
    
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("Please run: pip install insightface onnxruntime")

    _app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=det_size)
    return _app

def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0: return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def load_encodings() -> Tuple[np.ndarray, List[str]]:
    """
    Loads encodings from disk. 
    If missing, attempts to download from Supabase first.
    """
    # 1. Auto-Recovery: If file missing, try to download from Cloud
    if not ENCODINGS_PATH.exists():
        print("⚠️ Encodings file missing locally. Attempting cloud sync...")
        try:
            # Lazy import to avoid circular dependency issues
            from app.utils.supabase_utils import download_encodings_from_supabase
            
            # We need to load env vars here to ensure connection works
            from dotenv import load_dotenv
            load_dotenv(APP_DIR.parent / "secrets.env")
            
            # Attempt download
            download_encodings_from_supabase(str(ENCODINGS_PATH))
        except Exception as e:
            print(f"❌ Cloud download failed: {e}")

    # 2. Load from Disk
    if not ENCODINGS_PATH.exists():
        return np.array([]), []

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        return normalize_encodings(np.array(data["encodings"])), [str(i) for i in data["ids"]]
    except Exception as e:
        print(f"Error loading encodings pickle: {e}")
        return np.array([]), []

def verify_face(img_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Optional[dict]:
    known_encs, known_ids = load_encodings()
    
    if known_encs.size == 0:
        return {"status": "error", "message": "Database empty. Please run Sync in Admin Panel."}

    app = get_insightface()
    faces = app.get(img_bgr)
    
    if not faces:
        return None

    # Get largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)

    # Calculate distances
    dists = 1.0 - np.dot(known_encs, captured_emb)
    idx = np.argmin(dists)
    score = float(1.0 - dists[idx])

    if dists[idx] < threshold:
        return {"status": "success", "student_id": known_ids[idx], "confidence": score}
    else:
        return {"status": "failed", "student_id": "Unknown", "confidence": score}