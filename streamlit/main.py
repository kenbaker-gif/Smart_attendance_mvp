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
# Add project root to sys.path so we can import 'app' and 'scripts'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# Load Secrets (Local development)
env_path = project_root / "secrets.env"
load_dotenv(env_path)

# Configuration
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase Client
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"⚠️ Database Connection Error: {e}")

# Import Internal Modules
try:
    from app.face_engine.insightface_engine import verify_face
    from scripts.streamlit_app_regen import generate_encodings
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.stop()

# -----------------------------
# 2. Helper Functions
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
        # Query: SELECT name FROM students WHERE id = '...'
        response = supabase.table("students") \
            .select("name") \
            .eq("id", student_id) \
            .execute()

        if response.data and len(response.data) > 0:
            return response.data[0]['name']
        
    except Exception as e:
        print(f"⚠️ Name fetch error: {e}")
    
    return student_id # Fallback to ID if name not found

def add_attendance_record(student_id: str, confidence: float, status: str):
    """
    Logs the attendance attempt to Supabase 'attendance_records'.
    """
    if not supabase:
        return

    # Skip logging "Unknown" people if you prefer to save space
    # if student_id == "Unknown": return

    data = {
        # If unknown, we can't link to a foreign key, so we handle that logic here.
        # Ideally, your DB allows NULL student_id for failed attempts, 
        # OR you only log successes.
        "student_id": student_id if status == "success" else None,
        "confidence": float(confidence),
        "detection_method": "insightface_buffalo_s",
        "verified": status
    }

    try:
        supabase.table('attendance_records').insert(data).execute()
        print(f"✅ Logged: {student_id} ({status})")
    except Exception as e:
        print(f"❌ Logging Failed: {e}")
        # If it's a specific FK error, print a helpful hint
        if "foreign key" in str(e).lower():
            print(f"   👉 HINT: Student ID '{student_id}' is not in the 'students' table.")

# -----------------------------
# 3. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸")
    st.title("📸 AI Attendance System")

    tab1, tab2 = st.tabs(["Scanning Station", "Admin Panel"])

    # --- Tab 1: Attendance ---
    with tab1:
        st.header("Face Verification")
        
        # Camera Input
        img_file = st.camera_input("Capture Face")

        if img_file:
            if st.button("Verify Identity", type="primary"):
                # Convert image
                img = Image.open(img_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                with st.spinner("Analyzing biometric data..."):
                    result = verify_face(img_bgr)

                # Handle Results
                if result is None:
                    st.warning("⚠️ No face detected. Please try again.")
                
                elif result.get("status") == "error":
                    st.error(result["message"])
                
                elif result["status"] == "success":
                    raw_id = result['student_id']
                    
                    # 1. Get Real Name
                    display_name = get_student_name(raw_id)
                    
                    # 2. UI Feedback
                    st.balloons()
                    st.success(f"✅ Verified: **{display_name}**")
                    st.caption(f"ID: {raw_id} | Confidence: {result['confidence']:.2f}")
                    
                    # 3. Log to DB
                    add_attendance_record(raw_id, result['confidence'], "success")
                    
                else:
                    st.error("⛔ Access Denied: Unknown Person")
                    st.info(f"Confidence: {result['confidence']:.2f}")
                    
                    # Log failure (Optional)
                    add_attendance_record("Unknown", result['confidence'], "failed")

    # --- Tab 2: Admin ---
    with tab2:
        st.header("Database Management")
        password = st.text_input("Admin Password", type="password")
        
        if password == ADMIN_SECRET:
            st.success("Access Granted")
            
            st.divider()
            st.subheader("🔄 System Sync")
            st.info("This process will:\n1. Download new images from Cloud\n2. Retrain the AI model\n3. Upload the new database")
            
            if st.button("Start Sync & Retrain"):
                # 1. Create Progress Bar
                progress_bar = st.progress(0, text="Initializing...")
                
                # 2. Define Callback for UI Updates
                def update_progress_ui(percent, message):
                    # Clamp value between 0.0 and 1.0
                    safe_percent = min(max(percent, 0.0), 1.0)
                    progress_bar.progress(safe_percent, text=message)
                
                # 3. Run Generator
                try:
                    success = generate_encodings(progress_callback=update_progress_ui)
                    
                    if success:
                        st.success("✅ System updated successfully!")
                        st.cache_resource.clear() # Clear cache to load new encodings immediately
                        st.rerun() # Refresh app
                    else:
                        st.error("❌ Update failed. Please check the terminal logs.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
        elif password:
            st.error("Invalid Password")

if __name__ == "__main__":
    main()