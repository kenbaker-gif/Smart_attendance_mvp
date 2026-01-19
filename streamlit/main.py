import streamlit as st
import numpy as np
from PIL import Image
import cv2
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# -----------------------------
# Path & Setup
# -----------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# Load Secrets
env_path = project_root / "secrets.env"
load_dotenv(env_path)

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Init Supabase (Safe Mode)
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Connect Error: {e}")

try:
    from app.face_engine.insightface_engine import verify_face
    from scripts.streamlit_app_regen import generate_encodings
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()
    
# -----------------------------
# ✅ NEW: Name Lookup Helper (Updated for models.py)
# -----------------------------
@st.cache_data(ttl=300) 
def get_student_name(student_id: str) -> str:
    """
    Fetches the real name from Supabase 'students' table.
    Matches schema: Student(id, name)
    """
    if not supabase:
        return student_id

    try:
        # Query matching your models.py structure
        response = supabase.table("students") \
            .select("name") \
            .eq("id", student_id) \
            .execute()

        # Check if we got a result
        if response.data and len(response.data) > 0:
            return response.data[0]['name']
        
    except Exception as e:
        print(f"Name fetch error: {e}")
    
    return student_id # Fallback to ID if name not found

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
                # ✅ NEW: Fetch the real name before displaying
                raw_id = result['student_id']
                display_name = get_student_name(raw_id)

                st.balloons()
                st.success(f"✅ Verified: **{display_name}**") # Shows Name
                st.caption(f"ID: {raw_id} | Confidence: {result['confidence']:.2f}")
                
            else:
                st.error("⛔ Access Denied: Unknown Person")
                st.info(f"Confidence: {result['confidence']:.2f}")

# --- Tab 2: Admin ---
with tab2:
    st.header("Database Management")
    password = st.text_input("Admin Password", type="password")
    
    if password == ADMIN_SECRET:
        st.success("Unlocked")
        st.write("### Sync Database")
        st.info("Syncs images from Cloud, retrains model, and uploads results.")
        
        if st.button("🔄 Regenerate Encodings"):
            with st.spinner("Syncing & Retraining..."):
                success = generate_encodings()
            
            if success:
                st.success("System updated successfully!")
                st.cache_resource.clear()
            else:
                st.error("Update failed. Check logs.")
    elif password:
        st.error("Wrong password")