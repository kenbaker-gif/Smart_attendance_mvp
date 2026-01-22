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
# GLOBAL MEMORY (RAM)
# -----------------------------
_app = None
_CACHE_ENCODINGS = np.array([])
_CACHE_IDS = []

def get_insightface(det_size=(640, 640), model_name="buffalo_s"):
    global _app
    if _app is not None:
        return _app
    
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("Please run: pip install insightface onnxruntime")

    # Initialize InsightFace
    _app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=det_size)
    return _app

def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0: return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

# --- NEW FUNCTION: Called by API to inject new data ---
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
        # 1. Separate Keys (IDs) and Values (Embeddings)
        ids = list(new_data_dict.keys())
        embeddings = list(new_data_dict.values())
        
        # 2. Convert to Numpy Array (float32 is faster)
        emb_array = np.array(embeddings, dtype=np.float32)
        
        # 3. Normalize immediately (Crucial for Cosine Similarity)
        _CACHE_ENCODINGS = normalize_encodings(emb_array)
        _CACHE_IDS = ids
        
        print(f"✅ Engine: Memory Updated! Holding {len(_CACHE_IDS)} students.")
        
    except Exception as e:
        print(f"❌ Engine Update Error: {e}")

def verify_face(img_bgr: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Optional[dict]:
    # 1. USE RAM CACHE INSTEAD OF DISK
    global _CACHE_ENCODINGS, _CACHE_IDS
    
    # Safety Check: Is memory empty?
    if _CACHE_ENCODINGS.size == 0:
        return {"status": "error", "message": "Server is warming up... Try again in 10s."}

    # 2. Get AI Model
    app = get_insightface()
    faces = app.get(img_bgr)
    
    if not faces:
        return None

    # 3. Get largest face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)

    # 4. Compare with Memory (Vectorized Math)
    # Cosine Similarity: 1.0 = Match, 0.0 = No Match
    dists = 1.0 - np.dot(_CACHE_ENCODINGS, captured_emb)
    idx = np.argmin(dists)     # Find index of smallest distance (best match)
    score = float(1.0 - dists[idx]) # Convert distance to confidence score

    # 5. Prepare Coordinates (for Flutter Overlay)
    # Convert numpy int32 to standard python int/list
    bbox = face.bbox.astype(int).tolist()
    kps = face.kps.astype(int).tolist()

    if dists[idx] < threshold:
        return {
            "status": "success", 
            "student_id": _CACHE_IDS[idx], 
            "confidence": score,
            "bbox": bbox,
            "kps": kps
        }
    else:
        return {
            "status": "failed", 
            "student_id": "Unknown", 
            "confidence": score,
            "bbox": bbox, # Still send box so we can draw RED overlay
            "kps": kps
        }