# ======================================
# streamlit/main.py
# ======================================

import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime

# -----------------------------
# Add project root to Python path
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# -----------------------------
# Import face engine
# -----------------------------
import app.face_engine.insightface_engine as fe
from app.database import add_attendance_record as db_add_attendance

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path(__file__).parent / "data"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "attendance.log"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin123")
LOG_COOLDOWN_SECONDS = 60

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if 'log_cache' not in st.session_state:
    st.session_state.log_cache = {}

# -----------------------------
# Local logging helper
# -----------------------------
def log_attendance(student_id, status, confidence=0.0):
    """Logs attendance locally and in memory cache."""
    now = datetime.now()
    last_logged = st.session_state.log_cache.get(student_id)
    if status == 'success' and last_logged and (now - last_logged).seconds < LOG_COOLDOWN_SECONDS:
        return

    # Cache to prevent double logging
    st.session_state.log_cache[student_id] = now

    # Local log file
    with open(LOG_FILE, "a") as f:
        f.write(f"{now} | {student_id} | {status} | {confidence:.2f}\n")

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        page_title="Smart Attendance System",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📸 Smart Attendance System")

    tab1, tab2 = st.tabs(["Verification", "Admin Panel"])

    # -------------------------
    # Load encodings
    # -------------------------
    known_encs, known_ids = fe.load_encodings()

    # -------------------------
    # Verification Tab
    # -------------------------
    with tab1:
        sid = st.text_input("Student ID")
        img_file = st.camera_input("Capture Face")

        if sid and img_file and st.button("Verify Identity"):
            img = Image.open(img_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces = fe.get_engine().get(img_bgr)

            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                result, conf = fe.verify_face(known_encs, known_ids, face.embedding, sid)

                if result:
                    st.success(f"Verified: {sid} ({conf:.2f})")
                    st.balloons()
                    log_attendance(sid, "success", conf)
                    record = db_add_attendance(
                        student_id=sid,
                        confidence=conf,
                        detection_method="face",
                        verified="success"
                    )
                else:
                    st.error("Access Denied")
                    log_attendance(sid, "failed", conf)
                    record = db_add_attendance(
                        student_id=sid,
                        confidence=conf,
                        detection_method="face",
                        verified="failed"
                    )

                if record:
                    st.info(f"Attendance logged to DB for {sid}")
                else:
                    st.warning(f"⚠ Failed to log attendance to DB for {sid}")
            else:
                st.warning("No face detected")

    # -------------------------
    # Admin Panel Tab
    # -------------------------
    with tab2:
        admin_pass = st.text_input("Admin Secret", type="password")
        if admin_pass == ADMIN_SECRET:
            st.success("Access Granted")

            # Progress bar and label
            progress_bar = st.progress(0)
            progress_label = st.empty()

            def update_progress(msg, step=None, total=None):
                progress_label.text(msg)
                if step is not None and total is not None and total > 0:
                    progress = int((step / total) * 100)
                    progress_bar.progress(progress)

            # Regenerate Face Database button
            if st.button("Regenerate Face Database"):
                success = fe.generate_encodings(progress_callback=update_progress)
                if success:
                    st.success("✅ Face database regenerated")
                    known_encs, known_ids = fe.load_encodings()
                    st.experimental_refresh()
                else:
                    st.error("❌ Failed to generate encodings. Check raw_faces folder or Supabase config.")
        else:
            if admin_pass:
                st.error("Invalid Secret")


# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    main()
