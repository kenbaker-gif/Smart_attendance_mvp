# ==============================================================================
# FULL SCRIPT: Streamlit Face Recognition Attendance System
#
# Combines:
# 1. Supabase Initialization and Image Sync (Robust Download)
# 2. Face Encoding Generation (from recognition.py/encoding_utils.py)
# 3. Streamlit UI, Camera Verification, and Admin Panel
# 4. Integrated Logging (Console and Rotating File Log)
#
# Dependencies: streamlit, face_recognition, numpy, python-dotenv, supabase
# ==============================================================================

import os
import sys
import shutil
import pickle
import numpy as np
import streamlit as st
import face_recognition
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional, Union, Dict
from datetime import datetime, timedelta # Import timedelta for cache logic

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
# Set standard paths for data storage
IMAGES_DIR: Path = ROOT_DIR / "data" / "raw_faces"
ENCODINGS_PATH: Path = ROOT_DIR / "data" / "encodings_facenet.pkl"


# =============================================================
# --- LOGGING CONFIGURATION (From logger.py) ---
# =============================================================

LOG_DIR = CURRENT_DIR / "logs" # Use relative path from script location
LOG_FILE = LOG_DIR / "attendance.log"

# Create logs directory if missing
try:
    LOG_DIR.mkdir(exist_ok=True, parents=True) 
except Exception as e:
    # Print raw error if directory creation fails, as logging isn't fully set up yet.
    print(f"ERROR: Could not create log directory at {LOG_DIR}: {e}", file=sys.stderr)


# Configure main logger
logger = logging.getLogger("attendance_system")
logger.setLevel(logging.INFO) # Set to INFO for production use (DEBUG for troubleshooting)


# File Handler (rotates when 5MB)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO) 

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 


# Define Formatter and Attach
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logging configuration loaded successfully.")


# =============================================================
# --- ENVIRONMENT VARIABLE CONFIG ---
# =============================================================
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET: Optional[str] = os.getenv("SUPABASE_BUCKET")
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"

# Recognition threshold (distance) - lower is stricter
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
# Stores the last time a 'success' log was written for a student ID.
# This helps prevent log spam from rapid, successful verifications.
_log_cache: Dict[str, datetime] = {}
LOG_COOLDOWN_SECONDS = 60 # Time (seconds) to wait before logging a SUCCESS again

# =============================================================
# Database Interaction (Updated to use Supabase RDB)
# =============================================================
def add_attendance_record(student_id, confidence, model, status):
    """
    Logs attendance records to the Supabase 'attendance_records' table, matching the
    column names defined in the user's models.py schema, and applies local log
    deduplication for successful entries.
    """
    current_time = datetime.now()
    
    # --- DEDUPLICATION LOGIC ---
    if status == 'success':
        last_logged_time = _log_cache.get(student_id)
        if last_logged_time:
            time_since_last_log = current_time - last_logged_time
            if time_since_last_log < timedelta(seconds=LOG_COOLDOWN_SECONDS):
                logger.debug(f"DB log skipped for {student_id} (Success) due to cooldown.")
                # We still try to save to DB, but skip the logging.info() and toast
                # if the user only verifies the DB log, not the success toast.
                # However, in this app, we tie the toast to the DB operation.
                
                # To prevent unnecessary DB calls if the user rapidly clicks,
                # we could put the DB logic inside the cache check, but since
                # the DB is the source of truth, we ensure the DB call is made.
                pass # Continue to DB insertion, but skip local logging/toast check below
            
    # --- SUPABASE DB INSERTION ---
    
    logger.info(f"Attempting DB log for {student_id} with status {status}")
    
    if not USE_SUPABASE or not supabase:
        st.toast("📝 Attendance Log: Supabase is disabled. Record not saved.", icon="⚠️")
        logger.warning(f"DB log skipped for {student_id}: Supabase is disabled.")
        return

    try:
        # NOTE: Keys are mapped to match the column names in the AttendanceRecord model
        record = {
            "student_id": student_id,
            "confidence": float(confidence),
            "detection_method": model,  # Mapped from 'model' to 'detection_method'
            "verified": status,          # Mapped from 'status' to 'verified'
            "timestamp": datetime.now().isoformat() # Mapped from 'recorded_at' to 'timestamp'
        }

        # Attempt to insert into the 'attendance_records' table (matching models.py)
        response = supabase.table('attendance_records').insert(record).execute()
        
        # Check for immediate errors from the Supabase client
        if response.data and not response.data[0]:
             st.error(f"DB Insert Failed: No data returned. Check RLS/Table structure.")
             logger.error(f"DB Insert Failed for {student_id}: Supabase returned no data. Response: {response}")
             return

        # If DB insertion succeeded, update the local log cache for success status
        if status == 'success':
            _log_cache[student_id] = current_time
            
        st.toast(f"📝 DB: Logged **{status.upper()}** for **{student_id}** (Conf: {confidence:.2f})", icon="✅" if status == 'success' else "❌")
        logger.info(f"Successfully logged attendance for {student_id}. Conf: {confidence:.2f}, Status: {status}")
        
    except Exception as e:
        st.error(f"❌ DB Logging failed. Check Supabase connection/RLS policy. Error: {e}")
        logger.exception(f"FATAL ERROR during DB insertion for {student_id}.")


# =============================================================
# SUPABASE UTILITIES (Download Pipeline)
# =============================================================

def download_all_supabase_images(local_base_dir: Path):
# ... (rest of the download_all_supabase_images function remains unchanged)
# ...
    """Downloads all student images from Supabase to the local raw_faces directory using a robust two-stage approach (List Folders, then List/Download contents)."""
    if not USE_SUPABASE or not supabase or not SUPABASE_BUCKET:
        logger.info("Supabase download skipped (Disabled or client failed).")
        return True

    storage_api = supabase.storage.from_(SUPABASE_BUCKET)
    
    # 1. Clear the local directory completely before syncing
    if local_base_dir.exists():
        try:
            shutil.rmtree(local_base_dir)
            logger.info(f"🗑️ Cleared local directory: {local_base_dir}")
        except OSError as e:
            st.error(f"❌ Error clearing directory {local_base_dir}: {e}")
            logger.error(f"Error clearing directory {local_base_dir}: {e}")
            return False

    local_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. List all top-level items (folders/student IDs)
    try:
        root_entries = storage_api.list(path="")
    except Exception as e:
        st.error(f"❌ Supabase List Root Failed: {e}")
        logger.exception(f"Supabase List Root Failed.")
        return False

    prefixes_to_download = []
    for entry in root_entries:
        name = entry.get('name')
        if name and entry.get('id') is None:
            prefixes_to_download.append(name.strip('/'))

    if not prefixes_to_download:
        logger.warning("🛑 No student ID folders found in the bucket root.")
        return True

    logger.info(f"Found {len(prefixes_to_download)} folders to process: {prefixes_to_download}")
    
    total_downloaded = 0
    image_extensions = (".jpg", ".jpeg", ".png")

    with st.spinner(f"🔄 Syncing {len(prefixes_to_download)} student folders from Supabase..."):
        # 3. Iterate through each identified folder and download its content
        for prefix in prefixes_to_download:
            local_student_folder = local_base_dir / prefix
            local_student_folder.mkdir(exist_ok=True)
            
            try:
                # List files *inside* the current folder (prefix)
                file_entries = storage_api.list(path=prefix)
            except Exception as e:
                logger.error(f"❌ Failed to LIST contents for folder '{prefix}': {e}")
                continue

            for file_entry in file_entries:
                file_name = file_entry.get('name')
                
                if not file_name or not Path(file_name).suffix.lower() in image_extensions:
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
                        else:
                            logger.error(f"❌ Downloaded 0 bytes for {remote_path}. Check RLS/Permissions.")
                    elif isinstance(data, dict) and data.get("error"):
                         logger.error(f"❌ Download failed for {remote_path} (Supabase error: {data.get('error')})")
                    
                except Exception as e:
                    logger.error(f"❌ Download Failed for path {remote_path}: {type(e).__name__} - {e}")
                    
    st.success(f"🗃️ Synced {total_downloaded} files across {len(prefixes_to_download)} folders from Supabase.")
    logger.info(f"Completed sync. {total_downloaded} files downloaded.")
    return True

# =============================================================
# ENCODING GENERATION UTILITIES
# =============================================================

def generate_encodings(images_dir, output_path):
# ... (rest of the generate_encodings function remains unchanged)
# ...
    """Generates face encodings from local images and saves them to a pickle file."""
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        
    # 1. Download from Supabase first
    if USE_SUPABASE:
        download_all_supabase_images(images_dir)

    encodings = []
    ids = []
    processed_count = 0
    skipped_count = 0

    with st.spinner("🧠 Generating face encodings from local images..."):
        for student_folder in images_dir.iterdir():
            if not student_folder.is_dir():
                continue

            sid = student_folder.name
            for img_file in student_folder.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                try:
                    img = face_recognition.load_image_file(str(img_file))
                    boxes = face_recognition.face_locations(img, model="hog")
                    
                    if len(boxes) != 1: # Only encode images with exactly one face
                        skipped_count += 1
                        logger.debug(f"Skipping {img_file}: Found {len(boxes)} faces (must be 1).")
                        continue
                        
                    # Calculate encoding
                    enc = face_recognition.face_encodings(img, boxes)[0]
                    encodings.append(enc)
                    ids.append(sid)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing image {img_file}: {e}")
                    skipped_count += 1
                    continue

    if len(encodings) == 0:
        st.warning("⚠️ No valid faces found to generate encodings.")
        logger.warning("No valid faces found to generate encodings.")
        return False

    data = {
        "encodings": np.array(encodings),
        "ids": np.array(ids)
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        st.success(f"💾 Saved {processed_count} encodings for {len(set(ids))} students.")
        st.caption(f"Skipped {skipped_count} images (no face or multiple faces).")
        logger.info(f"Successfully saved {processed_count} encodings. Skipped: {skipped_count}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save encodings: {e}")
        logger.exception("Failed to save encodings to pickle file.")
        return False


# =============================================================
# RECOGNITION HELPERS
# =============================================================

def _safe_get(data_dict, *keys):
# ... (rest of the _safe_get function remains unchanged)
# ...
    """Safely extract data from a dictionary, handling multiple possible keys."""
    for k in keys:
        if k in data_dict and data_dict[k] is not None:
            return data_dict[k]
    return None

def _to_list(value):
# ... (rest of the _to_list function remains unchanged)
# ...
    """Converts various array/list types to a standard Python list."""
    if value is None: return []
    if isinstance(value, (list, tuple)): return list(value)
    if hasattr(value, "tolist"):
        try: return value.tolist()
        except: pass
    if isinstance(value, np.ndarray): return value.tolist()
    return [value]


def compare_128dim_encodings(known, uploaded, threshold=0.6):
# ... (rest of the compare_128dim_encodings function remains unchanged)
# ...
    """Compare 128-dimensional encodings (dlib/face_recognition default)."""
    distances = face_recognition.face_distance(known, uploaded)
    min_d = np.min(distances)
    idx = np.argmin(distances)
    is_match = min_d < threshold
    return min_d, idx, is_match

def compare_512dim_encodings(known, uploaded, threshold=0.5):
# ... (rest of the compare_512dim_encodings function remains unchanged)
# ...
    """Compare 512-dimensional encodings (FaceNet/ArcFace compatible) using L2 norm."""
    # Ensure uploaded is 2D for broadcasting with known encodings
    uploaded = np.array(uploaded).reshape(1, -1)
    
    # Calculate Euclidean (L2) distance
    distances = np.linalg.norm(np.array(known) - uploaded, axis=1) 
    min_d = np.min(distances)
    idx = np.argmin(distances)
    is_match = min_d < threshold
    return min_d, idx, is_match

# =============================================================
# ENCODING LOADER (Cached)
# =============================================================

@st.cache_resource
def load_encodings():
# ... (rest of the load_encodings function remains unchanged)
# ...
    """
    Loads encodings from pickle. Triggers sync/generation if file is missing 
    or if sync is enabled.
    """
    logger.info("Starting load_encodings process.")
    
    if not ENCODINGS_PATH.exists():
        st.info("📂 Encodings file missing. Attempting to generate from local images...")
        success = generate_encodings(IMAGES_DIR, ENCODINGS_PATH)
        if not success:
            logger.error("Failed to generate encodings.")
            return [], [], 0
            
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            
        encs_raw = _safe_get(data, "encodings", "encodings_facenet")
        ids_raw = _safe_get(data, "ids", "labels")
        
        known_encodings = _to_list(encs_raw)
        known_ids = [str(i) for i in _to_list(ids_raw)]
        
        if not known_encodings:
             st.warning("⚠️ Loaded file but found no encodings inside. Try retraining.")
             logger.warning("Loaded file but found no encodings inside.")
             return [], [], 0
             
        encoding_dim = len(known_encodings[0])
        
        st.success(f"✅ Loaded {len(known_ids)} encodings for {len(set(known_ids))} students ({encoding_dim}-dim)")
        logger.info(f"Successfully loaded {len(known_ids)} encodings ({encoding_dim}-dim)")
        return known_encodings, known_ids, encoding_dim
    except Exception as e:
        st.error(f"❌ Error loading encodings: {e}")
        logger.exception("FATAL ERROR: Failed to load encodings from pickle file.")
        return [], [], 0

# =============================================================
# MAIN STREAMLIT APPLICATION
# =============================================================

def main():
    st.set_page_config(page_title="Smart Attendance System", layout="centered")
    st.title("📸 Smart Attendance - Camera Verification")

    # Load data globally for the session, leveraging the cache
    known_encodings, known_ids, encoding_dim = load_encodings()
    
    # Determine the threshold based on the loaded encoding dimension
    threshold = 0.6 if encoding_dim == 128 else 0.5
    
    if known_encodings:
        st.info(f"System Ready. Loaded {len(set(known_ids))} unique students. (Threshold: {threshold})")
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

    # --- VERIFICATION LOGIC ---
    if camera_input:
        st.divider()
        image = Image.open(camera_input).convert("RGB")
        img_array = np.array(image)
        st.image(image, caption="Captured Image")

        # Face detection
        boxes = face_recognition.face_locations(img_array, model="hog") 
        used_model = "HOG" 

        if len(boxes) == 0:
            st.error("❌ No face detected. Improve lighting and visibility.")
            logger.warning("No face detected in captured image.")
            return
        elif len(boxes) > 1:
            st.warning(f"Multiple faces detected ({len(boxes)}). Focusing on the first one.")
            logger.warning(f"Multiple faces detected ({len(boxes)}).")
            
        try:
            # Encode the first detected face
            uploaded_encoding = face_recognition.face_encodings(img_array, boxes)[0]
        except Exception as e:
            st.error(f"Encoding failed: {e}")
            logger.error(f"Encoding failed after detection: {e}")
            uploaded_encoding = None

        if uploaded_encoding is not None and st.button("✅ Verify Identity"):
            if not student_id_input:
                st.error("Please enter your Student ID before verifying.")
                logger.warning("Verification attempted without Student ID.")
            elif not known_encodings:
                st.error("Cannot verify: No known encodings loaded.")
                logger.error("Verification failed: No known encodings available.")
            else:
                logger.info(f"Starting verification for {student_id_input}.")
                st.subheader("🔍 Verification Results")

                # Perform comparison
                if encoding_dim == 128:
                    min_d, idx, is_match = compare_128dim_encodings(known_encodings, uploaded_encoding, threshold)
                    logger.debug(f"Using 128-dim comparison. Min Distance: {min_d:.3f}")
                elif encoding_dim == 512:
                    min_d, idx, is_match = compare_512dim_encodings(known_encodings, uploaded_encoding, threshold)
                    logger.debug(f"Using 512-dim comparison. Min Distance: {min_d:.3f}")
                else:
                    st.error(f"Unsupported encoding dimension: {encoding_dim}. Cannot verify.")
                    logger.error(f"Unsupported encoding dimension: {encoding_dim}.")
                    return

                # Calculate confidence
                confidence = max(0, round(1 - (min_d / threshold), 3))
                matched_id = known_ids[idx]
                
                st.info(f"Nearest Match Found: **{matched_id}** (Distance: {min_d:.3f}, Threshold: {threshold})")

                if is_match and matched_id == student_id_input:
                    st.success(f"VERIFIED: **{student_id_input}**\nConfidence: {confidence*100:.1f}%")
                    st.balloons()
                    add_attendance_record(student_id_input, float(confidence), used_model, "success")
                else:
                    if is_match:
                        st.error(f"❌ Identity Mismatch. Recognized as **{matched_id}**, but entered ID is **{student_id_input}**.")
                        logger.warning(f"Mismatch: Entered {student_id_input}, Recognized {matched_id}.")
                    else:
                        st.error(f"❌ Verification failed. Face is too different (Distance {min_d:.3f} > {threshold}).")
                        logger.warning(f"Verification failed for {student_id_input}. Distance {min_d:.3f} > {threshold}.")

                    add_attendance_record(student_id_input, float(confidence), used_model, "failed")

    # --- ADMIN PANEL ---
    st.divider()
    with st.expander("🔧 Admin Panel"):
        col_admin1, col_admin2 = st.columns(2)
        with col_admin1:
            st.metric("Known Encoded Faces", len(known_ids))
            st.metric("Known Students (Unique IDs)", len(set(known_ids)))
        with col_admin2:
            st.metric("Encoding Dim", encoding_dim)
            st.metric("Threshold", threshold)
            st.metric("Supabase Sync", "Enabled" if USE_SUPABASE else "Disabled")

        if st.button("🔄 Retrain Encodings (Sync & Re-process)", key="retrain_btn"):
            st.info("Retraining encodings...")
            logger.info("Admin triggered: Retraining encodings.")
            # 1. Clear the cache to force a new execution of load_encodings logic
            load_encodings.clear() 
            
            # 2. Rerun the app to load new data
            st.rerun() 


if __name__ == "__main__":
    main()