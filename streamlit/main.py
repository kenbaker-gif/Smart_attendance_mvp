import streamlit as st
import numpy as np
from PIL import Image
import cv2
import sys
import os
from pathlib import Path
from dotenv import load_dotenv  # Import dotenv

# -----------------------------
# Critical Path Setup
# -----------------------------
# Add project root to sys.path to see 'app' and 'scripts'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# --- Load Secrets ---
# We look for secrets.env in the project root
env_path = project_root / "secrets.env"
load_dotenv(env_path)

# Get the password from env, default to "admin" if not set
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin")

try:
    from app.face_engine.insightface_engine import verify_face
    from scripts.streamlit_app_regen import generate_encodings
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Smart Attendance", page_icon="📸")

st.title("📸 AI Attendance System")

tab1, tab2 = st.tabs(["Scanning Station", "Admin Panel"])

# --- Tab 1: Attendance ---
with tab1:
    st.header("Face Verification")
    img_file = st.camera_input("Capture Face")

    if img_file:
        if st.button("Verify Identity"):
            img = Image.open(img_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            with st.spinner("Processing..."):
                result = verify_face(img_bgr)

            if result is None:
                st.warning("⚠️ No face detected.")
            elif result.get("status") == "error":
                st.error(result["message"])
            elif result["status"] == "success":
                st.balloons()
                st.success(f"✅ Verified: **{result['student_id']}**")
                st.info(f"Confidence: {result['confidence']:.2f}")
            else:
                st.error("⛔ Access Denied: Unknown Person")
                st.info(f"Confidence: {result['confidence']:.2f}")

# --- Tab 2: Admin ---
with tab2:
    st.header("Database Management")
    password = st.text_input("Admin Password", type="password")
    
    # ✅ Updated Check: Compares against the env variable
    if password == ADMIN_SECRET:
        st.success("Unlocked")
        st.write("### Sync Database")
        st.info("Scans `streamlit/data/raw_faces` and updates the engine.")
        
        if st.button("🔄 Regenerate Encodings"):
            with st.spinner("Scanning..."):
                success = generate_encodings()
            
            if success:
                st.success("Database updated successfully!")
                st.cache_resource.clear()
            else:
                st.error("Failed to update database. Check logs.")
    elif password:
        st.error("Wrong password")