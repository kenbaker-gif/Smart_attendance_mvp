import os
import sys
import gc
import pickle
import numpy as np
import streamlit as st
from streamlit.components.v1 import html
from pathlib import Path
from PIL import Image
from typing import List, Optional
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import cv2
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Project root
# -----------------------------
ABSOLUTE_PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(ABSOLUTE_PROJECT_ROOT))

RAW_FACES_DIR = ABSOLUTE_PROJECT_ROOT / "streamlit" / "data" / "raw_faces"
ENCODINGS_PATH = ABSOLUTE_PROJECT_ROOT / "streamlit" / "data" / "encodings_insightface.pkl"
TEMP_CAPTURE_PATH = ABSOLUTE_PROJECT_ROOT / "streamlit" / "data" / "tmp_capture.jpg"

INSIGHTFACE_MODEL_NAME = "buffalo_s"   # ✅ SAFE MODEL
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# -----------------------------
# Logging
# -----------------------------
LOG_DIR = ABSOLUTE_PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"
LOG_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger("attendance_system")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------
# InsightFace
# -----------------------------
from insightface.app import FaceAnalysis

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

@st.cache_resource
def init_insightface():
    logger.info("Initializing InsightFace (buffalo_s)")
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

_app_instance = None

def get_insightface_app():
    global _app_instance
    if _app_instance is None:
        _app_instance = init_insightface()
    return _app_instance

# -----------------------------
# Utilities
# -----------------------------
def normalize_encodings(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)

def _get_image_paths(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")]

def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

# -----------------------------
# Anti-spoofing
# -----------------------------
def is_image_blurry(image_rgb, threshold=80):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def face_too_small(face, min_ratio=0.15):
    x1, y1, x2, y2 = face.bbox
    face_area = (x2 - x1) * (y2 - y1)
    h, w = face.image_shape[:2]
    return face_area / (h * w) < min_ratio

# -----------------------------
# Face encoding
# -----------------------------
def extract_embedding(image_path: Path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if is_image_blurry(img_rgb):
        logger.warning("Rejected: blurry image")
        return None

    app = get_insightface_app()
    faces = app.get(img_rgb)
    if not faces:
        return None

    face = largest_face(faces)

    if face_too_small(face):
        logger.warning("Rejected: face too small")
        return None

    return np.array(face.embedding, dtype=np.float32)

# -----------------------------
# Generate encodings
# -----------------------------
def generate_encodings():
    RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

    encodings, ids = [], []

    for student_dir in RAW_FACES_DIR.iterdir():
        if not student_dir.is_dir():
            continue

        images = _get_image_paths(student_dir)
        success = 0

        for img in images:
            emb = extract_embedding(img)
            if emb is not None:
                encodings.append(emb)
                ids.append(student_dir.name)
                success += 1

        logger.info(f"{student_dir.name}: {success}/{len(images)} valid")

    if not encodings:
        return False

    encodings = normalize_encodings(np.array(encodings))
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": encodings, "ids": ids}, f)

    return True

# -----------------------------
# Load encodings
# -----------------------------
@st.cache_resource
def load_encodings():
    if not ENCODINGS_PATH.exists():
        generate_encodings()

    if not ENCODINGS_PATH.exists():
        return np.array([]), [], 0

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    enc = data["encodings"]
    ids = data["ids"]
    return enc, ids, enc.shape[1]

# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config("Smart Attendance", layout="centered")
    st.title("📸 Smart Attendance System")

    known_encodings, known_ids, encoding_dim = load_encodings()
    threshold = DEFAULT_THRESHOLD

    st.info(f"Students loaded: {len(set(known_ids))}")

    student_id = st.text_input("Student ID")

    image = st.camera_input("Capture image")

    if image:
        img = Image.open(image).convert("RGB")
        TEMP_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        img.save(TEMP_CAPTURE_PATH)

        emb = extract_embedding(TEMP_CAPTURE_PATH)
        os.remove(TEMP_CAPTURE_PATH)

        if emb is None:
            st.error("Face rejected (spoof / poor quality)")
            return

        emb /= np.linalg.norm(emb) + 1e-10
        dists = 1 - np.dot(known_encodings, emb)
        idx = np.argmin(dists)

        if dists[idx] <= threshold and known_ids[idx] == student_id:
            st.success("✅ VERIFIED")
            st.balloons()
        else:
            st.error("❌ NOT VERIFIED")

    with st.expander("🔧 Admin Panel"):
        st.metric("Known Faces", known_encodings.shape[0])
        st.metric("Unique Students", len(set(known_ids)))
        st.metric("Encoding Dim", encoding_dim)

        if st.button("🔄 Regenerate Encodings"):
            load_encodings.clear()
            generate_encodings()
            st.rerun()

if __name__ == "__main__":
    main()
