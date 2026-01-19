import os
import gc
import pickle
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENCODINGS_PATH = PROJECT_ROOT / "streamlit" / "data" / "encodings_insightface.pkl"

# -----------------------------
# Supabase (optional)
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "")

supabase = None
if USE_SUPABASE:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"⚠ Supabase init failed: {e}")

# -----------------------------
# InsightFace Engine
# -----------------------------
_engine = None
INSIGHTFACE_MODEL_NAME = "buffalo_s"

def get_engine(det_size=(320, 320)):
    global _engine
    if _engine is not None:
        return _engine

    from insightface.app import FaceAnalysis
    _engine = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    _engine.prepare(ctx_id=-1, det_size=det_size)
    print("✅ InsightFace engine ready")
    return _engine

# -----------------------------
# Encodings
# -----------------------------
def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0: return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def load_encodings():
    if ENCODINGS_PATH.exists():
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        return normalize_encodings(np.array(data["encodings"])), list(data["ids"])
    return np.array([]), []

def generate_encodings(progress_callback=None):
    """
    Generate face encodings for all students.
    Skips images that are already encoded.
    """
    if not USE_SUPABASE or not supabase:
        print("Supabase not configured. Generating from local images only.")

    from app.utils.supabase_utils import download_all_supabase_images

    RAW_FACES_DIR = PROJECT_ROOT / "streamlit" / "data" / "raw_faces"
    RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

    # Download new images only if Supabase is enabled
    if USE_SUPABASE and supabase:
        download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(RAW_FACES_DIR))

    # Load previously saved encodings
    known_encs, known_ids = load_encodings()
    processed_images = set()  # Keep track of already processed images

    if ENCODINGS_PATH.exists():
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        for sid, _ in zip(data["ids"], data["encodings"]):
            processed_images.add(sid)  # Mark student IDs already encoded

    encodings, ids = list(known_encs), list(known_ids)
    engine = get_engine()
    student_dirs = sorted([d for d in RAW_FACES_DIR.iterdir() if d.is_dir()])
    total_students = len(student_dirs)

    for idx, student_dir in enumerate(student_dirs, 1):
        student_id = student_dir.name
        if student_id in processed_images:
            if progress_callback:
                progress_callback(f"Skipping already encoded student {idx}/{total_students}: {student_id}")
            continue

        if progress_callback:
            progress_callback(f"Processing student {idx}/{total_students}: {student_id}")

        images = [img for img in student_dir.iterdir() if img.suffix.lower() in (".jpg", ".jpeg", ".png")]

        for img_idx, img_path in enumerate(images, 1):
            if progress_callback:
                progress_callback(f"  Image {img_idx}/{len(images)}: {img_path.name}")

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            faces = engine.get(img_bgr)
            if not faces:
                continue

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            encodings.append(face.embedding)
            ids.append(student_id)

    if not encodings:
        print("❌ No new faces detected")
        return False

    arr = normalize_encodings(np.array(encodings, dtype=np.float32))
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": arr, "ids": ids}, f)

    if progress_callback:
        progress_callback(f"✅ Saved {len(encodings)} encodings for {len(set(ids))} students")

    return True

# -----------------------------
# Verify face
# -----------------------------
def verify_face(known_encs, known_ids, face_embedding, student_id, threshold=0.5):
    """
    Compare a single face embedding against known encodings for a specific student ID.

    Returns:
        (bool, float): (is_match, confidence)
    """
    if known_encs.size == 0:
        return False, 0.0

    # Normalize the input face embedding
    face_embedding = face_embedding / (np.linalg.norm(face_embedding) + 1e-10)

    # Compute cosine distance
    dists = 1.0 - np.dot(known_encs, face_embedding)
    idx = np.argmin(dists)
    conf = float(1.0 - dists[idx])

    # Match if distance below threshold AND student ID matches
    if dists[idx] < threshold and str(known_ids[idx]).strip() == str(student_id).strip():
        return True, conf
    return False, conf
