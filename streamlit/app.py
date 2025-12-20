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
# Store data under the 'data' subfolder of this streamlit package (avoid duplicating 'streamlit' in the path)
DATA_DIR = PROJECT_ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
TEMP_CAPTURE_PATH = DATA_DIR / "tmp_capture.jpg"
SETTINGS_PATH = DATA_DIR / "settings.json"

# Default for auto-generation can be overridden with env var AUTO_GENERATE_ENCODINGS
AUTO_GENERATE_ENV = os.getenv("AUTO_GENERATE_ENCODINGS", "false").lower() == "true"

def _read_settings():
    try:
        import json
        if SETTINGS_PATH.exists():
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        logger.debug("Failed to read settings.json")
    return {}


def _write_settings(settings: dict):
    try:
        import json
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh)
        return True
    except Exception as e:
        logger.exception(f"Failed to write settings: {e}")
        return False


def _get_setting(key: str, default=None):
    s = _read_settings()
    return s.get(key, default)


def _set_setting(key: str, value):
    s = _read_settings()
    s[key] = value
    return _write_settings(s)

INSIGHTFACE_MODEL_NAME = "buffalo_s"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# Ensure writable directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Supabase config
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
            print("⚠ Supabase client not initialized: URL or KEY missing.")
            USE_SUPABASE = False
    except Exception as e:
        print(f"❌ Supabase init failed: {e}")
        USE_SUPABASE = False

    # Safe import of supabase_utils
    try:
        from app.utils.supabase_utils import download_all_supabase_images
    except Exception as e:
        print(f"⚠ Could not import supabase_utils. Supabase downloads disabled: {e}")
        download_all_supabase_images = None
        # Disable Supabase usage to avoid exposing a non-functional UI path
        USE_SUPABASE = False

# -----------------------------
# Logging
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
# InsightFace setup (lazy)
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    st.error("❌ ERROR: insightface not found. Install: pip install insightface[onnx]")
    st.stop()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Module-level placeholders. `app` is intentionally set to None so importing
# this module does not initialize the heavy model. Call `get_insightface()`
# at runtime to obtain the singleton instance (cached by Streamlit).
_app = None
app = None

@st.cache_resource(show_spinner=False)
def init_insightface(name: str = INSIGHTFACE_MODEL_NAME, det_size=(640, 640)):
    logger.info(f"Initializing InsightFace: {name} (CPU)")
    logger.debug(f"InsightFace det_size={det_size}")
    _local = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
    _local.prepare(ctx_id=-1, det_size=det_size)
    logger.info("InsightFace ready.")
    return _local

def get_insightface(det_size=(640, 640), name: str = INSIGHTFACE_MODEL_NAME):
    """Return the singleton InsightFace `FaceAnalysis` instance.

    This defers heavy initialization until needed and reuses the cached
    instance returned by `init_insightface()` (Streamlit's `st.cache_resource`).
    Other modules can request a smaller `det_size` by passing the
    preferred value; if the instance already exists it will be reused.
    """
    global _app, app
    logger.debug(f"get_insightface called with det_size={det_size} name={name}")
    # Warn about large detection sizes which can increase memory usage
    try:
        w, h = det_size
        if w * h > 640 * 640:
            logger.warning("Requested large det_size may increase memory usage: %s", det_size)
    except Exception:
        pass

    if _app is not None:
        return _app
    _app = init_insightface(name=name, det_size=det_size)
    app = _app
    return _app

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
        faces = get_insightface().get(img_rgb)
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
# Generate/load encodings
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    images_dir.mkdir(parents=True, exist_ok=True)
    if USE_SUPABASE and download_all_supabase_images and SUPABASE_URL and SUPABASE_KEY and SUPABASE_BUCKET:
        print("📦 Downloading images from Supabase...")
        ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=False)
        print("✅ Supabase download complete." if ok else "⚠️ Download failed or empty.")

    encodings, ids = [], []
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    for student_dir in student_dirs:
        student_id = student_dir.name
        image_paths = _get_image_paths(student_dir)
        for img_path in image_paths:
            emb = _generate_face_encoding_from_image(img_path)
            if emb is not None:
                encodings.append(emb)
                ids.append(student_id)

    if not encodings:
        logger.error("No encodings generated.")
        # Log available image folders/files for diagnostics
        try:
            student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
            logger.debug(f"Student folders: {[d.name for d in student_dirs]}")
            sample_files = []
            for d in student_dirs[:5]:
                files = [str(p.name) for p in _get_image_paths(d)][:5]
                sample_files.append({d.name: files})
            logger.debug(f"Sample files per folder (up to 5): {sample_files}")
        except Exception:
            logger.debug("Could not list image directories for diagnostics.")
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

@st.cache_resource
def load_encodings():
    # Determine whether we should auto-generate encodings when missing.
    auto_gen_setting = _get_setting("auto_generate_encodings", None)
    auto_generate = AUTO_GENERATE_ENV or (auto_gen_setting is True)

    if not ENCODINGS_PATH.exists():
        if auto_generate:
            st.info("Encodings missing. Auto-generating from images...")
            ok = generate_encodings(RAW_FACES_DIR, ENCODINGS_PATH)
            if not ok or not ENCODINGS_PATH.exists():
                msg = (
                    "No encodings found after auto-generation. Ensure that `streamlit/data/raw_faces` contains "
                    "student image folders (10-digit student IDs) or configure Supabase via env vars."
                )
                logger.error(msg)
                try:
                    st.warning(msg)
                except Exception:
                    pass
                return np.array([]), [], 0
        else:
            msg = (
                "Encodings are missing. Use the Admin Panel to generate encodings or enable auto-generation."
            )
            logger.warning(msg)
            try:
                st.info(msg)
            except Exception:
                pass
            return np.array([]), [], 0

    try:
        with open(ENCODINGS_PATH, "rb") as fh:
            data = pickle.load(fh)
        known_encodings = normalize_encodings(np.array(_to_list(data.get("encodings", [])), dtype=np.float32))
        known_ids = [str(i) for i in _to_list(data.get("ids", []))]
        return known_encodings, known_ids, known_encodings.shape[1] if known_encodings.size > 0 else 0
    except Exception as e:
        logger.exception("Failed to load encodings.")
        try:
            st.error("Failed to load encodings. See logs for details.")
        except Exception:
            pass
        return np.array([]), [], 0

# -----------------------------
# Attendance logging
# -----------------------------
_log_cache = {}
LOG_COOLDOWN_SECONDS = 60

def add_attendance_record(student_id: str, confidence: float, model: str, status: str):
    current_time = datetime.now()
    if status == "success":
        last = _log_cache.get(student_id)
        if last and (current_time - last).total_seconds() < LOG_COOLDOWN_SECONDS:
            return
    if not USE_SUPABASE or supabase is None:
        st.toast("Supabase disabled. Attendance not saved.", icon="⚠️")
        return
    try:
        record = {
            "student_id": student_id,
            "confidence": float(confidence),
            "detection_method": model,
            "verified": status,
            "timestamp": current_time.isoformat()
        }
        response = supabase.table("attendance_records").insert(record).execute()
        if hasattr(response, "data") and response.data:
            _log_cache[student_id] = current_time
            st.toast(f"Attendance logged for {student_id}", icon="✅")
        elif hasattr(response, "error") and response.error:
            logger.error(f"Supabase insertion failed: {response.error}")
        else:
            logger.warning(f"Unexpected Supabase response: {response}")
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
    st.info(f"System Ready: {len(set(known_ids))} students loaded. Threshold: {threshold}")

    student_id_input = st.text_input("Enter Student ID").strip()
    camera_input = st.camera_input("Capture Image")
    uploaded_embedding = None

    if camera_input:
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image")
        TEMP_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
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

        uploaded_embedding /= (np.linalg.norm(uploaded_embedding) + 1e-10)
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

        # Persistent auto-generation toggle
        auto_gen_setting = _get_setting("auto_generate_encodings", False)
        auto_gen = st.checkbox("Auto-generate encodings when missing (persisted)", value=auto_gen_setting)
        if auto_gen != auto_gen_setting:
            _set_setting("auto_generate_encodings", bool(auto_gen))
            st.success(f"Auto-generate set to {auto_gen}.")

        # Generate using local images
        if st.button("🔄 Generate Encodings Now (local images)"):
            st.info("Generating encodings from local images...")
            ok = generate_encodings(RAW_FACES_DIR, ENCODINGS_PATH)
            if ok:
                load_encodings.clear()
                st.success("Encodings generated successfully.")
            else:
                st.error("Failed to generate encodings. Check logs.")
            st.experimental_rerun()

        # Generate using Supabase (downloads then generates)
        if USE_SUPABASE:
            if not callable(download_all_supabase_images):
                st.error("Supabase downloads are disabled or not available. Check server logs and ensure `app/utils/supabase_utils.py` and the `supabase` package are installed.")
            else:
                if st.button("⬇️ Download from Supabase and Generate (clear local)"):
                    st.info("Downloading images from Supabase (clearing local images)...")
                    try:
                        ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(RAW_FACES_DIR), clear_local=True)
                        if ok:
                            st.info("Download complete. Generating encodings...")
                            ok2 = generate_encodings(RAW_FACES_DIR, ENCODINGS_PATH)
                            if ok2:
                                load_encodings.clear()
                                st.success("Encodings generated from Supabase images.")
                            else:
                                st.error("Failed to generate encodings after Supabase download.")
                        else:
                            st.error("Supabase download failed. Check logs and credentials.")
                    except Exception as e:
                        logger.exception("Supabase download and generate failed.")
                        try:
                            st.error(f"Supabase download failed: {e}")
                        except Exception:
                            pass
                    st.rerun()

        # Quick retrain shortcut (keeps existing behavior)
        if st.button("♻️ Retrain Encodings (force)"):
            st.info("Regenerating encodings from available images...")
            load_encodings.clear()
            init_insightface.clear()
            st.cache_resource.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
