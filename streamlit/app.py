# ==============================================================================
# FULL SCRIPT: Streamlit Face Recognition Attendance System
#
#
#
# Combines:
# 1. Supabase Initialization and Image Sync (Robust Download)
# 2. Face Encoding Generation (DeepFace/Facenet512, 512-dim)
# 3. Streamlit UI, Camera Verification, and Admin Panel
# 4. Integrated Logging (Console and Rotating File Log)
#
# Dependencies: streamlit, numpy, python-dotenv, supabase, deepface
# ==============================================================================

import os
import sys
import shutil
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Suppress verbose TensorFlow/C++ warnings related to memory allocation
# Setting to '3' means only FATAL errors will be printed.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- DEEPFACE IMPORT (Mandatory) ---
try:
    from deepface import DeepFace
except ImportError:
    st.error("❌ The 'deepface' package is not installed. Run: pip install deepface.")
    st.stop() # Stop execution if DeepFace is missing

# --- LOGGING IMPORTS ---
import logging
from logging.handlers import RotatingFileHandler

# Try to load environment variables from a .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # .env handling is optional

# -----------------------------
# --- Project Path Setup ------
# -----------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR

# --- DEEPFACE CONFIGURATION ---
MODEL_NAME = "Facenet512" # ALIGNMENT: Using Facenet512 for consistency with encoding_generator.py
TARGET_DIM = 512
ENCODINGS_PATH: Path = ROOT_DIR / "data" / f"encodings_{MODEL_NAME.lower()}.pkl"
IMAGES_DIR: Path = ROOT_DIR / "data" / "raw_faces"

# Default detector for all DeepFace operations (efficient and accurate)
DETECTOR_BACKEND = 'retinaface'


# =============================================================
# --- LOGGING CONFIGURATION ---
# =============================================================

LOG_DIR = CURRENT_DIR / "logs"
LOG_FILE = LOG_DIR / "attendance.log"

try:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
except Exception as e:
    print(f"ERROR: Could not create log directory at {LOG_DIR}: {e}", file=sys.stderr)


logger = logging.getLogger("attendance_system")
logger.setLevel(logging.DEBUG) # Changed to DEBUG

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Clear existing handlers to prevent duplicates on rerun
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logging configuration loaded successfully.")


# =============================================================
# --- ENVIRONMENT VARIABLE CONFIG ---
# =============================================================
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "")
# Default to false if not explicitly "true"
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"

# FaceNet standard threshold for L2 distance (lower is stricter).
# Increased from 0.5 to 0.6 to allow for minor variations in real-world capture.
DEFAULT_THRESHOLD = 0.6

# --- SUPABASE CLIENT GLOBAL STATE ---
supabase = None

if USE_SUPABASE:
    try:
        from supabase import create_client
        if SUPABASE_URL and SUPABASE_KEY and SUPABASE_BUCKET:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("✅ Supabase client initialized (Global)")
        else:
            logger.warning("⚠ Supabase client not initialized: URL, KEY, or BUCKET missing.")
            USE_SUPABASE = False
    except ImportError:
        st.error("❌ The 'supabase' package is not installed. Run: pip install supabase.")
        logger.error("The 'supabase' package is not installed.")
        USE_SUPABASE = False
    except Exception as e:
        st.error(f"❌ Failed to initialize Supabase client: {e}")
        logger.critical(f"Failed to initialize Supabase client: {e}")
        USE_SUPABASE = False


# =============================================================
# --- LOG DEDUPLICATION CACHE ---
# =============================================================
_log_cache: Dict[str, datetime] = {}
LOG_COOLDOWN_SECONDS = 60 # Time (seconds) to wait before logging a SUCCESS again

# =============================================================
# Database Interaction
# =============================================================
def add_attendance_record(student_id: str, confidence: float, model: str, status: str):
    """
    Logs attendance records to the Supabase 'attendance_records' table, and applies
    local log deduplication for successful entries.
    """
    current_time = datetime.now()

    # --- DEDUPLICATION LOGIC ---
    if status == 'success':
        last_logged_time = _log_cache.get(student_id)
        if last_logged_time and (current_time - last_logged_time) < timedelta(seconds=LOG_COOLDOWN_SECONDS):
            logger.debug(f"DB log skipped for {student_id} (Success) due to cooldown.")
            return # Skip if successful and within cooldown

    # --- SUPABASE DB INSERTION ---

    logger.info(f"Attempting DB log for {student_id} with status {status}")

    if not USE_SUPABASE or not supabase:
        st.toast("📝 Attendance Log: Supabase is disabled. Record not saved.", icon="⚠️")
        logger.warning(f"DB log skipped for {student_id}: Supabase is disabled.")
        return

    try:
        record = {
            "student_id": student_id,
            "confidence": float(confidence),
            "detection_method": model,
            "verified": status,
            "timestamp": datetime.now().isoformat()
        }

        response = supabase.table('attendance_records').insert(record).execute()

        if response.data and response.data[0]:
            # If DB insertion succeeded, update the local log cache for success status
            if status == 'success':
                _log_cache[student_id] = current_time

            st.toast(f"📝 DB: Logged **{status.upper()}** for **{student_id}** (Conf: {confidence:.2f})", icon="✅" if status == 'success' else "❌")
            logger.info(f"Successfully logged attendance for {student_id}. Conf: {confidence:.2f}, Status: {status}")
        else:
            # Handle cases where the response structure is unexpected or empty
            st.error(f"DB Insert Failed: Check RLS/Table structure. Response: {response}")
            logger.error(f"DB Insert Failed for {student_id}: Supabase returned no data. Response: {response}")

    except Exception as e:
        st.error(f"❌ DB Logging failed. Check Supabase connection/RLS policy. Error: {e}")
        logger.exception(f"FATAL ERROR during DB insertion for {student_id}.")


# =============================================================
# SUPABASE UTILITIES (Download Pipeline)
# =============================================================

def download_all_supabase_images(local_base_dir: Path):
    """Downloads all student images from Supabase to the local raw_faces directory."""
    if not USE_SUPABASE or not supabase or not SUPABASE_BUCKET:
        logger.info("Supabase download skipped (Disabled or client failed).")
        return True

    storage_api = supabase.storage.from_(SUPABASE_BUCKET)

    if local_base_dir.exists():
        try:
            shutil.rmtree(local_base_dir)
            logger.info(f"🗑️ Cleared local directory: {local_base_dir}")
        except OSError as e:
            st.error(f"❌ Error clearing directory {local_base_dir}: {e}")
            logger.error(f"Error clearing directory {local_base_dir}: {e}")
            return False

    local_base_dir.mkdir(parents=True, exist_ok=True)

    try:
        root_entries = storage_api.list(path="")
    except Exception as e:
        st.error(f"❌ Supabase List Root Failed: {e}")
        logger.exception(f"Supabase List Root Failed.")
        return False

    prefixes_to_download = [
        entry.get('name').strip('/')
        for entry in root_entries
        if entry.get('name') and entry.get('id') is None # Check if it's a folder/prefix
    ]

    if not prefixes_to_download:
        logger.warning("🛑 No student ID folders found in the bucket root.")
        return True

    logger.info(f"Found {len(prefixes_to_download)} folders to process: {prefixes_to_download}")

    total_downloaded = 0
    image_extensions = (".jpg", ".jpeg", ".png")

    with st.spinner(f"🔄 Syncing {len(prefixes_to_download)} student folders from Supabase..."):
        for prefix in prefixes_to_download:
            local_student_folder = local_base_dir / prefix
            local_student_folder.mkdir(exist_ok=True)

            try:
                file_entries = storage_api.list(path=prefix)
            except Exception as e:
                logger.error(f"❌ Failed to LIST contents for folder '{prefix}': {e}")
                continue

            for file_entry in file_entries:
                file_name = file_entry.get('name')

                if not file_name or Path(file_name).suffix.lower() not in image_extensions:
                    continue

                remote_path = f"{prefix}/{file_name}"
                local_file_path = local_student_folder / file_name

                try:
                    data = storage_api.download(remote_path)

                    if isinstance(data, bytes) and data:
                        with open(local_file_path, "wb") as f:
                            f.write(data)

                        if local_file_path.stat().st_size > 0:
                            total_downloaded += 1
                    elif isinstance(data, dict) and data.get("error"):
                          logger.error(f"❌ Download failed for {remote_path} (Supabase error: {data.get('error')})")

                except Exception as e:
                    logger.error(f"❌ Download Failed for path {remote_path}: {type(e).__name__} - {e}")

    st.success(f"🗃️ Synced {total_downloaded} files across {len(prefixes_to_download)} folders from Supabase.")
    logger.info(f"Completed sync. {total_downloaded} files downloaded.")
    return True

# =============================================================
# ENCODING GENERATION UTILITIES (DEEPFACE)
# =============================================================
def generate_encodings(images_dir: Path, output_path: Path) -> bool:
    """Generates face encodings using DeepFace (Facenet512) from local images and saves them."""
    
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
    
    if USE_SUPABASE:
        download_all_supabase_images(images_dir)

    encodings: List[np.ndarray] = []
    ids: List[str] = []
    processed_count = 0
    skipped_count = 0

    with st.spinner(f"🧠 Generating {MODEL_NAME} encodings from local images..."):
        for student_folder in images_dir.iterdir():
            if not student_folder.is_dir():
                continue

            sid = student_folder.name
            for img_file in student_folder.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                try:
                    # Use DeepFace.represent to generate embedding
                    # ALIGNMENT: Using DETECTOR_BACKEND (retinaface) consistently
                    representations = DeepFace.represent(
                        img_path=str(img_file), 
                        model_name=MODEL_NAME, 
                        enforce_detection=True, 
                        detector_backend=DETECTOR_BACKEND,
                        align=True
                    )
                    
                    if len(representations) != 1: 
                        skipped_count += 1
                        logger.debug(f"Skipping {img_file}: Found {len(representations)} faces (must be 1).")
                        continue
                        
                    # Extract the 512-dim embedding
                    enc = np.array(representations[0]["embedding"], dtype=np.float32)
                    
                    encodings.append(enc)
                    ids.append(sid)
                    processed_count += 1
                except ValueError as ve:
                    # DeepFace raises ValueError if enforce_detection=True and no face is found
                    skipped_count += 1
                    logger.warning(f"Skipping {img_file} for {sid}: No face detected (ValueError: {ve})")
                    continue
                except Exception as e:
                    skipped_count += 1
                    logger.error(f"ERROR processing {img_file} for {sid}: {type(e).__name__} - {e}")
                    continue

    # --- SAVE PICKLE ---
    if not encodings:
        st.error("❌ No encodings generated. Check images and log for errors.")
        logger.error("No encodings generated. Output file not saved.")
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            # Normalize all encodings before saving to ensure consistent L2 distance metrics
            "encodings": normalize_encodings(np.array(encodings, dtype=np.float32)), 
            "ids": np.array(ids)
        }

        with open(output_path, "wb") as fh:
            pickle.dump(data, fh)

        st.success(f"✅ Saved {len(encodings)} encodings for {len(set(ids))} students. (All L2 Normalized)")
        logger.info(f"Successfully saved {len(encodings)} encodings to {output_path}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save encodings: {e}")
        logger.exception("FATAL ERROR: Failed to save encodings to pickle file.")
        return False
        
# =============================================================
# RECOGNITION HELPERS (Pure NumPy)
# =============================================================

def _to_list(value: Any) -> List[Any]:
    """Converts various array/list types to a standard Python list."""
    if value is None: return []
    if isinstance(value, (list, tuple)): return list(value)
    if hasattr(value, "tolist"):
        try: return value.tolist()
        except: pass
    if isinstance(value, np.ndarray): return value.tolist()
    return [value]


def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    """Performs L2 normalization on a batch of vectors (rows)."""
    # Calculate the L2 norm (magnitude) for each vector
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero (for zero vectors)
    norms[norms == 0] = 1
    # Perform element-wise division to normalize
    return vectors / norms


def calculate_l2_distances(known_encodings: np.ndarray, uploaded_encoding: np.ndarray) -> tuple[float, int]:
    """
    Calculates the L2 (Euclidean) distance between the uploaded face encoding and
    all known encodings. Returns the minimum distance and the index of the closest match.
    """
    # Ensure uploaded encoding is a 2D array for correct NumPy broadcasting
    uploaded_encoding = uploaded_encoding.reshape(1, -1)
    
    # CRITICAL FIX: Ensure the uploaded encoding is L2 normalized before comparison
    normalized_uploaded = normalize_encodings(uploaded_encoding)

    # Note: known_encodings are assumed to be normalized by load_encodings,
    # but the distance function works fine if they are (as they should be).
    
    # Calculate Euclidean (L2) distance using NumPy's norm function
    # np.linalg.norm(A - B, axis=1) calculates the distance for each row in A against B
    distances = np.linalg.norm(known_encodings - normalized_uploaded, axis=1)

    min_d = np.min(distances)
    idx = np.argmin(distances)

    return min_d, idx


# =============================================================
# ENCODING LOADER (Cached)
# =============================================================

@st.cache_resource
def load_encodings():
    """
    Loads encodings from pickle. Triggers sync/generation if file is missing.
    """
    logger.info("Starting load_encodings process.")

    if not ENCODINGS_PATH.exists():
        st.info(f"📂 Encodings file missing ({MODEL_NAME}). Attempting to generate from local images...")
        success = generate_encodings(IMAGES_DIR, ENCODINGS_PATH)
        if not success:
            logger.error("Failed to generate encodings.")
            return np.array([]), [], 0

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)

        # Ensure we can retrieve data from the common keys
        known_encodings_list = _to_list(data.get("encodings", data.get("encodings_facenet")))
        known_ids = [str(i) for i in _to_list(data.get("ids", data.get("labels")))]

        if not known_encodings_list:
              st.warning("⚠️ Loaded file but found no encodings inside. Try retraining.")
              logger.warning("Loaded file but found no encodings inside.")
              return np.array([]), [], 0

        known_encodings = np.array(known_encodings_list)
        
        # FIX: Ensure loaded encodings are normalized immediately after loading
        known_encodings = normalize_encodings(known_encodings)
        
        encoding_dim = known_encodings.shape[1]

        st.success(f"✅ Loaded {len(known_ids)} encodings for {len(set(known_ids))} students ({encoding_dim}-dim, Model: {MODEL_NAME})")
        logger.info(f"Successfully loaded {len(known_ids)} encodings ({encoding_dim}-dim, Model: {MODEL_NAME})")
        return known_encodings, known_ids, encoding_dim
    except Exception as e:
        st.error(f"❌ Error loading encodings: {e}")
        logger.exception("FATAL ERROR: Failed to load encodings from pickle file.")
        return np.array([]), [], 0

# =============================================================
# MAIN STREAMLIT APPLICATION
# =============================================================

def main():
    st.set_page_config(page_title="Smart Attendance System", layout="centered")
    st.title("📸 Smart Attendance - Camera Verification (DeepFace)")

    # Load data globally for the session, leveraging the cache
    known_encodings, known_ids, encoding_dim = load_encodings()

    # Use the FaceNet standard threshold for 512-dim L2 distance
    threshold = DEFAULT_THRESHOLD

    if known_encodings.size > 0:
        st.info(f"System Ready. Loaded {len(set(known_ids))} unique students. (Model: {MODEL_NAME}, Threshold: {threshold})")
    else:
        st.warning("⚠️ No encodings loaded. Please use the Admin Panel to sync images and Retrain.")

    # --- UI LAYOUT ---
    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Capture Image")
        camera_input = st.camera_input("Point camera at your face")

    with col2:
        st.subheader("ℹ️ Instructions")
        st.info("""
        1. Look directly at camera
        2. Ensure good lighting
        3. Center your face
        4. Enter ID and click Verify
        """)

    student_id_input = st.text_input("Enter Student ID", placeholder="e.g., 2400102415", key="student_id")

    uploaded_encoding: Optional[np.ndarray] = None
    used_model = MODEL_NAME

    # Temp file path for DeepFace processing
    temp_path = CURRENT_DIR / "temp_capture.jpg"

    # --- ENCODING CAPTURED IMAGE ---
    if camera_input:
        st.divider()
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image")

        temp_path.parent.mkdir(exist_ok=True)
        image.save(temp_path)

        try:
            st.info(f"Detecting and encoding face using {MODEL_NAME}...")
            # ALIGNMENT: Using DETECTOR_BACKEND (retinaface) consistently
            representations = DeepFace.represent(
                img_path=str(temp_path),
                model_name=MODEL_NAME,
                enforce_detection=True,
                detector_backend=DETECTOR_BACKEND, 
            )

            if len(representations) == 0:
                st.error("❌ No face detected. Improve lighting/visibility.")
                logger.warning("No face detected by DeepFace in captured image.")
                uploaded_encoding = None
            elif len(representations) > 1:
                st.warning(f"Multiple faces detected ({len(representations)}). Using the largest detected face.")
                # DeepFace returns them sorted by size/confidence, so we use the first
                uploaded_encoding = np.array(representations[0]["embedding"], dtype=np.float32)
            else:
                uploaded_encoding = np.array(representations[0]["embedding"], dtype=np.float32)


        except ValueError as ve:
            # DeepFace raises ValueError if enforce_detection=True and no face is found
            st.error(f"❌ DeepFace Error (No Face Detected): Please center your face. ({ve})")
            logger.error(f"DeepFace encoding failed: {ve}")
            uploaded_encoding = None
        except Exception as e:
            st.error(f"❌ Encoding failed: {e}")
            logger.error(f"Encoding failed after detection: {e}")
            uploaded_encoding = None
        finally:
            if temp_path.exists():
                os.remove(temp_path)


        # --- VERIFICATION BUTTON ---
        if uploaded_encoding is not None and st.button("✅ Verify Identity"):
            if not student_id_input:
                st.error("Please enter your Student ID before verifying.")
                logger.warning("Verification attempted without Student ID.")
            elif known_encodings.size == 0:
                st.error("Cannot verify: No known encodings loaded.")
                logger.error("Verification failed: No known encodings available.")
            else:
                logger.info(f"Starting verification for {student_id_input}.")
                st.subheader("🔍 Verification Results")

                # Perform comparison using pure NumPy L2 distance
                min_d, idx = calculate_l2_distances(known_encodings, uploaded_encoding)

                is_match = min_d < threshold

                # Confidence calculation is based on the distance relative to the threshold
                # If distance is 0, confidence is 1.0. If distance is threshold, confidence is 0.
                confidence = max(0.0, round(1.0 - (min_d / threshold), 3))

                matched_id = known_ids[idx]

                st.info(f"Nearest Match Found: **{matched_id}** (Distance: {min_d:.3f}, Threshold: {threshold})")

                if is_match and matched_id == student_id_input:
                    st.success(f"VERIFIED: **{student_id_input}**\nConfidence: {confidence*100:.1f}%")
                    st.balloons()
                    add_attendance_record(student_id_input, confidence, used_model, "success")
                else:
                    status = "failed"
                    if is_match:
                        st.error(f"❌ Identity Mismatch. Recognized as **{matched_id}**, but entered ID is **{student_id_input}**.")
                        logger.warning(f"Mismatch: Entered {student_id_input}, Recognized {matched_id}.")
                        status = "mismatch"
                    else:
                        st.error(f"❌ Verification failed. Face is too different (Distance {min_d:.3f} > {threshold}).")
                        logger.warning(f"Verification failed for {student_id_input}. Distance {min_d:.3f} > {threshold}.")

                    add_attendance_record(student_id_input, confidence, used_model, status)

    # --- ADMIN PANEL ---
    st.divider()
    with st.expander("🔧 Admin Panel"):
        col_admin1, col_admin2 = st.columns(2)
        with col_admin1:
            st.metric("Known Encoded Faces", known_encodings.shape[0])
            st.metric("Known Students (Unique IDs)", len(set(known_ids)))
        with col_admin2:
            st.metric("Encoding Dim", encoding_dim)
            st.metric("Threshold (L2)", threshold)
            st.metric("Supabase Sync", "Enabled" if USE_SUPABASE else "Disabled")
            st.caption(f"Feature Extractor: **{MODEL_NAME}**")

        if st.button("🔄 Retrain Encodings (Sync & Re-process)", key="retrain_btn"):
            st.info("Retraining encodings...")
            logger.info("Admin triggered: Retraining encodings.")

            # Clear the cache to force a new execution of load_encodings logic
            load_encodings.clear()

            # Rerun the app to load new data
            st.rerun()


if __name__ == "__main__":
    main()