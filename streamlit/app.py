import os
import sys
import gc
import pickle
import numpy as np
import streamlit as st
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
# Project root and paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
TEMP_CAPTURE_PATH = DATA_DIR / "tmp_capture.jpg"
SETTINGS_PATH = DATA_DIR / "settings.json"

ENCODINGS_REMOTE_PATH = os.getenv("ENCODINGS_REMOTE_PATH", "encodings/encodings_insightface.pkl")
ENCODINGS_REMOTE_TYPE = os.getenv("ENCODINGS_REMOTE_TYPE", "supabase")

AUTO_GENERATE_ENV = os.getenv("AUTO_GENERATE_ENCODINGS", "false").lower() == "true"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logging Setup
# -----------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"
LOG_DIR.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger("attendance_system")
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -----------------------------
# Supabase Configuration
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "")

download_all_supabase_images = None
supabase = None

if USE_SUPABASE:
    try:
        from supabase import create_client
        if SUPABASE_URL and SUPABASE_KEY:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase client initialized")
        else:
            USE_SUPABASE = False
    except Exception as e:
        print(f"❌ Supabase init failed: {e}")
        USE_SUPABASE = False

    def _load_supabase_utils():
        try:
            # Try package import
            from app.utils.supabase_utils import download_all_supabase_images as fn
            return fn
        except Exception:
            try:
                # Fallback to file path import
                import importlib.util
                utils_path = PROJECT_ROOT.parent / "app" / "utils" / "supabase_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("project_supabase_utils", str(utils_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return getattr(mod, "download_all_supabase_images", None)
            except Exception:
                return None
        return None

    download_all_supabase_images = _load_supabase_utils()

# -----------------------------
# InsightFace Logic
# -----------------------------
INSIGHTFACE_MODEL_NAME = "buffalo_s"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    st.error("❌ insightface not found. Run: pip install insightface onnxruntime")
    st.stop()

@st.cache_resource(show_spinner=False)
def init_insightface(name=INSIGHTFACE_MODEL_NAME, det_size=(640, 640)):
    logger.info(f"Initializing InsightFace model: {name}")
    _local = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
    _local.prepare(ctx_id=-1, det_size=det_size)
    return _local

def get_insightface(det_size=(640, 640)):
    return init_insightface(det_size=det_size)

# -----------------------------
# Processing Utilities
# -----------------------------
def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0: return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def _generate_face_encoding_from_image(path: Path) -> Optional[np.ndarray]:
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None: return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = get_insightface().get(img_rgb)
        if not faces: return None
        # Find largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return np.array(face.embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Encoding error for {path.name}: {e}")
        return None

# -----------------------------
# Core: Load Encodings (with Cloud Sync)
# -----------------------------
@st.cache_resource
def load_encodings():
    """Tries to load local pkl, otherwise pulls from Supabase."""
    if not ENCODINGS_PATH.exists():
        if USE_SUPABASE and supabase:
            try:
                with st.spinner("☁️ Syncing encodings from Cloud..."):
                    res = supabase.storage.from_(SUPABASE_BUCKET).download(ENCODINGS_REMOTE_PATH)
                    # Handle Supabase response variations
                    data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
                    
                    if data_bytes:
                        with open(ENCODINGS_PATH, "wb") as f:
                            f.write(data_bytes)
                        logger.info("✅ Encodings downloaded from Supabase.")
            except Exception as e:
                logger.warning(f"Could not sync from cloud: {e}")

    if ENCODINGS_PATH.exists():
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            encs = np.array(data.get("encodings", []), dtype=np.float32)
            ids = [str(i) for i in data.get("ids", [])]
            return normalize_encodings(encs), ids, encs.shape[1] if encs.size > 0 else 0
        except Exception as e:
            logger.error(f"Error loading pkl: {e}")
            
    return np.array([]), [], 0

def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    """Manual trigger to generate pkl from images_dir."""
    encodings, ids = [], []
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    
    if not student_dirs:
        return False

    detector = get_insightface()
    for s_dir in student_dirs:
        for img_p in [p for p in s_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]:
            emb = _generate_face_encoding_from_image(img_p)
            if emb is not None:
                encodings.append(emb)
                ids.append(s_dir.name)

    if encodings:
        arr = normalize_encodings(np.array(encodings))
        with open(output_path, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        return True
    return False

# -----------------------------
# UI Components
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸")
    st.title("📸 Smart Attendance System")

    known_encs, known_ids, _ = load_encodings()

    if known_encs.size == 0:
        st.warning("⚠️ No student data found. Please sync from Admin Panel.")
    
    col1, col2 = st.columns(2)
    with col1:
        student_id_input = st.text_input("Student ID")
    with col2:
        camera_image = st.camera_input("Scan Face")

    if camera_image and student_id_input:
        img = Image.open(camera_image).convert("RGB")
        img.save(TEMP_CAPTURE_PATH)
        captured_emb = _generate_face_encoding_from_image(TEMP_CAPTURE_PATH)
        
        if captured_emb is not None:
            captured_emb /= (np.linalg.norm(captured_emb) + 1e-10)
            dists = 1.0 - np.dot(known_encs, captured_emb)
            best_idx = np.argmin(dists)
            
            if dists[best_idx] < DEFAULT_THRESHOLD and known_ids[best_idx] == student_id_input:
                st.success(f"Verified: {student_id_input}")
                st.balloons()
            else:
                st.error("Verification Failed: Identity mismatch or low confidence.")

    # Admin Panel
    with st.expander("🛠️ Admin Settings"):
        if st.button("🔄 Sync with Supabase Cloud"):
            if USE_SUPABASE and download_all_supabase_images:
                ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(RAW_FACES_DIR))
                if ok:
                    generate_encodings()
                    st.cache_resource.clear()
                    st.success("Synchronized successfully!")
                    st.rerun()

if __name__ == "__main__":
    main()