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
# 1. Path & Environment Setup
# -----------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
sys.path.append(str(project_root))

env_path = project_root / "secrets.env"
if env_path.exists():
    load_dotenv(env_path)

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"⚠️ Database Connection Error: {e}")

# Import Internal Modules
try:
    from app.face_engine.insightface_engine import verify_face
    # ✅ CORRECT IMPORT: Pointing to the main CI script
    from scripts.ci_generate_encodings import generate_encodings
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.stop()

# -----------------------------
# 2. Helper Functions
# -----------------------------
@st.cache_data(ttl=300) 
def get_student_name(student_id: str) -> str:
    if not supabase: return student_id
    try:
        response = supabase.table("students").select("name").eq("id", student_id).execute()
        if response.data: return response.data[0]['name']
    except Exception: pass
    return student_id 

def add_attendance_record(student_id: str, confidence: float, status: str):
    if not supabase: return
    data = {
        "student_id": student_id if status == "success" else None,
        "confidence": float(confidence),
        "detection_method": "web_dashboard",
        "verified": status
    }
    try:
        supabase.table('attendance_records').insert(data).execute()
    except Exception as e:
        print(f"❌ Logging Failed: {e}")

# -----------------------------
# 3. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸", layout="wide")
    st.title("📸 AI Attendance System")

    tab1, tab2 = st.tabs(["Scanning Station", "Admin Panel"])

    # --- Tab 1: Web-Based Attendance ---
    with tab1:
        st.header("Face Verification")
        img_file = st.camera_input("Capture Face")

        if img_file:
            if st.button("Verify Identity", type="primary"):
                img = Image.open(img_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                with st.spinner("Analyzing biometric data..."):
                    result = verify_face(img_bgr)

                if result is None:
                    st.warning("⚠️ No face detected. Please try again.")
                elif result.get("status") == "error":
                    st.error(result["message"])
                elif result["status"] == "success":
                    raw_id = result['student_id']
                    display_name = get_student_name(raw_id)
                    st.balloons()
                    st.success(f"✅ Verified: **{display_name}**")
                    add_attendance_record(raw_id, result['confidence'], "success")
                else:
                    st.error("⛔ Access Denied: Unknown Person")
                    add_attendance_record("Unknown", result['confidence'], "failed")

    # --- Tab 2: Admin Panel ---
    with tab2:
        st.header("Database Management")
        password = st.text_input("Admin Password", type="password")
        
        if password == ADMIN_SECRET:
            st.success("Access Granted")
            st.divider()
            st.subheader("🔄 System Sync")
            st.info("This will: 1. Sync Cloud Images | 2. Retrain AI | 3. Update DB")
            
            if st.button("Start Sync & Retrain"):
                progress_bar = st.progress(0, text="Initializing...")
                
                # ✅ WRAPPER FUNCTION: Matches what script expects (percent, text)
                def update_progress_ui(percent, message="Processing..."):
                    # Ensure percent is between 0.0 and 1.0
                    safe_percent = min(max(float(percent), 0.0), 1.0)
                    progress_bar.progress(safe_percent, text=message)
                
                try:
                    # Pass the wrapper to the script
                    success = generate_encodings(progress_callback=update_progress_ui)
                    
                    if success:
                        st.success("✅ System updated successfully!")
                        st.cache_resource.clear() 
                    else:
                        st.error("❌ Update failed. Check terminal logs.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif password:
            st.error("Invalid Password")

if __name__ == "__main__":
    main()