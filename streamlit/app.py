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
# Project root
# -----------------------------
ABSOLUTE_PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(ABSOLUTE_PROJECT_ROOT))

RAW_FACES_DIR: Path = ABSOLUTE_PROJECT_ROOT / "streamlit" / "data" / "raw_faces"
ENCODINGS_PATH: Path = ABSOLUTE_PROJECT_ROOT / "streamlit" / "data" / "encodings_insightface.pkl"
TEMP_CAPTURE_PATH: Path = ABSOLUTE_PROJECT_ROOT / "streamlit" / "data" / "tmp_capture.jpg"

INSIGHTFACE_MODEL_NAME = "buffalo_s"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# Supabase config
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "")

# -----------------------------
# Logging
# -----------------------------
LOG_DIR = ABSOLUTE_PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"
LOG_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger("attendance_system")
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -----------------------------
# Supabase integration
# -----------------------------
download_all_supabase_images = None
supabase = None

if USE_SUPABASE:
    try:
        from supabase import create_client
        if SUPABASE_URL and SUPABASE_KEY:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("✅ Supabase client initialized")
        else:
            logger.warning("⚠ Supabase client not initialized: URL or KEY missing.")
            USE_SUPABASE = False
    except Exception as e:
        logger.error(f"Supabase init failed: {e}")
        USE_SUPABASE = False

    # Safe import of supabase_utils
    try:
        from app.utils.supabase_utils import download_all_supabase_images
    except ImportError:
        logger.warning("⚠ Could not import supabase_utils. Supabase downloads disabled.")
        download_all_supabase_images = None

# -----------------------------
# InsightFace
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    st.error("❌ ERROR: insightface not found. Install with: pip install insightface[onnx]")
    st.stop()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@st.cache_resource(show_spinner=False)
def init_insightface():
    logger.info(f"Initializing InsightFace: {INSIGHTFACE_MODEL_NAME} (CPU)")
    app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    logger.info("InsightFace ready.")
    return app

app = init_insightface()

# -----------------------------
# Utilities
# -----------------------------
def _to_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]

def _get_image_paths(student_dir: Path) -> List[Path]:
    return sorted([p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")])

def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def _largest_face(faces):
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = map(float, f.bbox)
        return max(0.0, (x2 - x1)*(y2 - y1))
    return max(faces, key=area)

def _generate_face_encoding_from_image(path: Path) -> Optional[np.ndarray]:
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            logger.warning(f"Failed to read image {path}")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if not faces:
            logger.info(f"No face detected in {path.name}")
            return None
        face = _largest_face(faces)
        if face is None or getattr(face, "embedding", None) is None:
            return None
        emb = np.array(face.embedding, dtype=np.float32)
        del img_bgr, img_rgb, faces, face
        gc.collect()
        return emb
    except Exception as e:
        logger.exception(f"InsightFace error for {path.name}: {e}")
        gc.collect()
        return None

# -----------------------------
# Generate encodings
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    images_dir.mkdir(parents=True, exist_ok=True)
    if USE_SUPABASE and download_all_supabase_images and SUPABASE_URL and SUPABASE_KEY and SUPABASE_BUCKET:
        logger.info("📦 Downloading images from Supabase...")
        ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=False)
        logger.info("✅ Supabase download complete." if ok else "⚠️ Download failed or empty.")

    encodings, ids = [], []
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    logger.info(f"Found {len(student_dirs)} student folders to process.")

    for student_dir in student_dirs:
        student_id = student_dir.name
        image_paths = _get_image_paths(student_dir)
        if not image_paths:
            logger.info(f"No images for {student_id}, skipping.")
            continue
        logger.info(f"Processing {student_id} ({len(image_paths)} images)...")
        for img_path in image_paths:
            emb = _generate_face_encoding_from_image(img_path)
            if emb is None:
                continue
            encodings.append(emb)
            ids.append(student_id)

    if not encodings:
        logger.error("No encodings generated.")
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(output_path, "wb") as fh:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, fh)
        logger.info(f"Saved {len(encodings)} encodings for {len(set(ids))} students → {output_path}")
        return True
    except Exception as e:
        logger.exception(f"Failed to save encodings: {e}")
        return False

# -----------------------------
# Load encodings
# -----------------------------
@st.cache_resource
def load_encodings():
    if not ENCODINGS_PATH.exists():
        st.info("Encodings missing. Generating from images...")
        ok = generate_encodings(RAW_FACES_DIR, ENCODINGS_PATH)
        if not ok:
            logger.error("Failed to generate encodings.")
            return np.array([]), [], 0
    try:
        with open(ENCODINGS_PATH, "rb") as fh:
            data = pickle.load(fh)
        known_encodings = normalize_encodings(np.array(_to_list(data.get("encodings", [])), dtype=np.float32))
        known_ids = [str(i) for i in _to_list(data.get("ids", []))]
        return known_encodings, known_ids, known_encodings.shape[1] if known_encodings.size > 0 else 0
    except Exception as e:
        logger.exception("Failed to load encodings.")
        return np.array([]), [], 0

# -----------------------------
# Attendance logging
# -----------------------------
_log_cache = {}
LOG_COOLDOWN_SECONDS = 60

def add_attendance_record(student_id: str, confidence: float, model: str, status: str):
    current_time = datetime.now()
    
    # Prevent repeated logging within cooldown
    if status == "success":
        last = _log_cache.get(student_id)
        if last and (current_time - last).total_seconds() < LOG_COOLDOWN_SECONDS:
            return

    if not USE_SUPABASE or supabase is None:
        st.toast("Supabase disabled. Attendance not saved.", icon="⚠️")
        logger.warning(f"DB log skipped for {student_id}: Supabase disabled.")
        return

    try:
        record = {
            "student_id": student_id,
            "confidence": float(confidence),
            "detection_method": model,
            "verified": status,
            "timestamp": datetime.now().isoformat()
        }

        response = supabase.table("attendance_records").insert(record).execute()

        # --- Corrected logic ---
        # The SDK returns .data on success, .error on failure (may be None)
        if hasattr(response, "data") and response.data:
            _log_cache[student_id] = current_time
            st.toast(f"Attendance logged for {student_id}", icon="✅")
            logger.info(f"Attendance logged for {student_id}: {response.data}")
        elif hasattr(response, "error") and response.error:
            logger.error(f"Supabase insertion failed: {response.error}")
        else:
            # Catch-all fallback
            logger.warning(f"Supabase insertion returned unexpected response: {response}")

    except Exception as e:
        logger.exception("DB insertion failed.")


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", layout="centered")
    st.title("📸 Smart Attendance System (InsightFace)")

    known_encodings, known_ids, encoding_dim = load_encodings()
    threshold = DEFAULT_THRESHOLD

    st.info(f"System Ready: {len(set(known_ids))} students loaded. (Model: {INSIGHTFACE_MODEL_NAME}, Threshold: {threshold})")

    student_id_input = st.text_input("Enter Student ID", placeholder="e.g., 2400102415").strip()
    camera_input = st.camera_input("Capture Image")

    uploaded_embedding = None

    if camera_input:
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image")
        TEMP_CAPTURE_PATH.parent.mkdir(exist_ok=True, parents=True)
        image.save(TEMP_CAPTURE_PATH)
        uploaded_embedding = _generate_face_encoding_from_image(TEMP_CAPTURE_PATH)
        if TEMP_CAPTURE_PATH.exists():
            os.remove(TEMP_CAPTURE_PATH)

    if uploaded_embedding is not None and st.button("✅ Verify Identity"):
        if not student_id_input:
            st.error("Enter Student ID")
            return
        if known_encodings.size == 0:
            st.error("No known encodings loaded")
            return

        uploaded_embedding = uploaded_embedding / (np.linalg.norm(uploaded_embedding)+1e-10)
        dists = 1.0 - np.dot(known_encodings, uploaded_embedding)
        idx = int(np.argmin(dists))
        min_d = float(dists[idx])
        confidence = 1.0 - min_d
        matched_id = known_ids[idx]

        if min_d <= threshold and matched_id == student_id_input:
            st.success(f"VERIFIED: {student_id_input} (Confidence: {confidence*100:.1f}%)")
            st.balloons()
            add_attendance_record(student_id_input, confidence, INSIGHTFACE_MODEL_NAME, "success")
        else:
            st.error(f"❌ Verification failed. Matched {matched_id}, Distance {min_d:.3f}")
            add_attendance_record(student_id_input, confidence, INSIGHTFACE_MODEL_NAME, "failed")

    # Admin panel
    with st.expander("🔧 Admin Panel"):
        st.metric("Known Faces", known_encodings.shape[0])
        st.metric("Unique Students", len(set(known_ids)))
        st.metric("Encoding Dim", encoding_dim)
        st.metric("Threshold", threshold)
        st.metric("Supabase Sync", "Enabled" if USE_SUPABASE else "Disabled")
        if st.button("🔄 Retrain Encodings"):
            st.info("Regenerating encodings...")
            load_encodings.clear()
            init_insightface.clear()
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()
