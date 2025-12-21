import os
import sys
import gc
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import cv2
from dotenv import load_dotenv
from pydantic import BaseModel
import importlib.util

# -----------------------------
# Configuration & Paths
# -----------------------------
load_dotenv()
# Resolve the root whether running from root or inside /streamlit
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"

# Ensure directories exist
for d in [DATA_DIR, RAW_FACES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INSIGHTFACE_MODEL_NAME = "buffalo_s"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
ENCODINGS_REMOTE_PATH = os.getenv("ENCODINGS_REMOTE_PATH", "raw_faces/encodings/encodings_insightface.pkl")

# -----------------------------
# Logging Setup
# -----------------------------
logger = logging.getLogger("attendance_system")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

# -----------------------------
# Supabase & State
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "")

supabase = None
download_all_supabase_images = None

# Bulletproof Utility Loading
def load_supabase_utils():
    try:
        util_path = PROJECT_ROOT / "app" / "utils" / "supabase_utils.py"
        if util_path.exists():
            spec = importlib.util.spec_from_file_location("supabase_utils", util_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.download_all_supabase_images
    except Exception as e:
        logger.error(f"Utility load error: {e}")
    return None

if USE_SUPABASE:
    try:
        from supabase import create_client
        if SUPABASE_URL and SUPABASE_KEY:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        download_all_supabase_images = load_supabase_utils()
    except Exception as e:
        logger.error(f"Supabase init error: {e}")

# Permanent Schema
class AttendanceRecordIn(BaseModel):
    student_id: str
    confidence: float
    detection_method: str
    verified: str

# -----------------------------
# InsightFace Engine
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ImportError:
    if "--generate" not in sys.argv:
        st.error("❌ insightface not found.")
        st.stop()
    else:
        print("❌ Error: insightface not installed.")
        sys.exit(1)

@st.cache_resource(show_spinner=False)
def get_insightface(det_size=(640, 640)):
    _app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=det_size)
    return _app

# -----------------------------
# Core Utilities
# -----------------------------
def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0: return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def add_attendance_record(student_id: str, confidence: float, status: str):
    current_time = datetime.now()
    if 'log_cache' not in st.session_state: st.session_state.log_cache = {}
    
    if status == 'success':
        last_logged = st.session_state.log_cache.get(student_id)
        if last_logged and (current_time - last_logged) < timedelta(seconds=60):
            return

    logger.info(f"Log: {student_id} | Status: {status} | Conf: {confidence:.2f}")

    if USE_SUPABASE and supabase:
        try:
            record = AttendanceRecordIn(
                student_id=student_id,
                confidence=float(confidence),
                detection_method="insightface_buffalo_s",
                verified=status
            )
            supabase.table('attendance_records').insert(record.model_dump()).execute()
            if status == 'success':
                st.session_state.log_cache[student_id] = current_time
            st.toast(f"Synced: {student_id}", icon="✅")
        except Exception as e:
            logger.error(f"DB Sync Error: {e}")

# -----------------------------
# Process Pipeline
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    print(f"📂 Scanning directory: {images_dir}")
    
    # 1. Force download images from Supabase first
    if USE_SUPABASE and download_all_supabase_images:
        print("🔄 Downloading images from Supabase...")
        download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=True)
    
    # 2. Check for student folders
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    if not student_dirs:
        print("⚠️ No student directories found in raw_faces!")
        return False

    encodings, ids = [], []
    engine = get_insightface()

    for s_dir in student_dirs:
        student_id = s_dir.name
        img_files = [p for p in s_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        for img_p in img_files:
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None: continue
            faces = engine.get(img_bgr)
            if faces:
                # Get largest face
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                encodings.append(face.embedding)
                ids.append(student_id)
                print(f"✅ Encoded: {student_id}")

    if not encodings:
        print("❌ No valid faces found in images.")
        return False

    try:
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(output_path, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        print(f"🎉 Success! Generated {len(ids)} encodings.")
        return True
    except Exception as e:
        print(f"❌ Pickle error: {e}")
        return False

@st.cache_resource
def load_encodings():
    if not ENCODINGS_PATH.exists() and USE_SUPABASE and supabase:
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(ENCODINGS_REMOTE_PATH)
            # Handle different return types from Supabase
            data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
            if data_bytes:
                with open(ENCODINGS_PATH, "wb") as f: f.write(data_bytes)
        except Exception as e:
            logger.error(f"Cloud download fail: {e}")

    if ENCODINGS_PATH.exists():
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        return normalize_encodings(np.array(data["encodings"])), [str(i) for i in data["ids"]]
    return np.array([]), []

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸", layout="wide")
    st.title("📸 Biometric Attendance System")

    known_encs, known_ids = load_encodings()
    tab1, tab2 = st.tabs(["🎯 Verification", "⚙️ Admin Panel"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            sid = st.text_input("Enter Student ID", placeholder="e.g. 24001")
            img_file = st.camera_input("Capture Face")
        
        with col2:
            if sid and img_file and st.button("Verify Identity", use_container_width=True):
                img = Image.open(img_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                faces = get_insightface().get(img_bgr)
                
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(img_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Analysis Result")

                    # Match Logic
                    captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
                    if known_encs.size > 0:
                        dists = 1.0 - np.dot(known_encs, captured_emb)
                        idx = np.argmin(dists)
                        conf = float(1.0 - dists[idx])
                        
                        if dists[idx] < DEFAULT_THRESHOLD and known_ids[idx] == sid:
                            st.success(f"WELCOME {sid}! (Match: {conf*100:.1f}%)")
                            st.balloons()
                            add_attendance_record(sid, conf, "success")
                        else:
                            st.error(f"Verification Failed. Unauthorized access attempt.")
                            add_attendance_record(sid, conf, "failed")
                    else:
                        st.error("System Database Empty. Contact Admin.")
                else:
                    st.warning("No face detected. Please adjust lighting.")

    with tab2:
        st.subheader("📊 System Management")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Sync Cloud Data", use_container_width=True):
                with st.spinner("Syncing..."):
                    if generate_encodings():
                        st.success("Database Updated!")
                        st.rerun()
        with c2:
            if LOG_FILE.exists():
                with open(LOG_FILE, "r") as f:
                    st.download_button("📥 Download Logs", f.read(), "attendance.log", use_container_width=True)
        with c3:
            if st.button("🗑️ Reset Local Cache", use_container_width=True):
                if ENCODINGS_PATH.exists(): os.remove(ENCODINGS_PATH)
                st.rerun()

if __name__ == "__main__":
    if "--generate" in sys.argv:
        print("🚀 Executing Automated Encoding Generation...")
        success = generate_encodings()
        sys.exit(0 if success else 4)
    else:
        main()