import streamlit as st
import os
import time
from supabase import create_client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "student_faces")

# Fix for trailing slash error
if URL and not URL.endswith("/"):
    URL += "/"

if not URL or not KEY:
    st.error("❌ Missing credentials in .env")
    st.stop()

supabase = create_client(URL, KEY)

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="Face Capture Pro", page_icon="📸", layout="wide")
st.title("📸 Student Face Capture")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Setup")
    # We assign keys to these widgets so we can "kill" them later to reset the UI
    student_id = st.text_input("Enter Student ID", key="id_box").strip()
    camera_image = st.camera_input("2. Take Photo", key="cam_box")

# --- 3. PREVIEW & LOGIC ---
if camera_image:
    with col2:
        st.subheader("2. Review & Save")
        st.image(camera_image, caption="Capture Preview", width='stretch')
        
        if not student_id:
            st.warning("⚠️ Enter Student ID to enable upload.")
        else:
            if st.button("🚀 Upload & Clear Everything", width='stretch'):
                try:
                    # A. Determine sequence
                    res = supabase.storage.from_(BUCKET).list(student_id)
                    existing_files = [f['name'] for f in (res or []) if f['name'].lower().endswith('.jpg')]
                    next_num = len(existing_files) + 1
                    
                    # B. Upload Loop (Collision Protection)
                    uploaded = False
                    while not uploaded:
                        file_name = f"{next_num}.jpg"
                        path = f"{student_id}/{file_name}"
                        try:
                            supabase.storage.from_(BUCKET).upload(
                                path=path,
                                file=camera_image.getvalue(),
                                file_options={"content-type": "image/jpeg"}
                            )
                            uploaded = True
                        except Exception as e:
                            if "409" in str(e) or "already exists" in str(e).lower():
                                next_num += 1
                            else: raise e

                    st.success(f"✅ Successfully saved as {file_name}")
                    st.toast("Clearing form for next capture...")

                    # --- C. THE FULL RESET ---
                    # This removes the image and the text from the app's memory
                    if "cam_box" in st.session_state:
                        del st.session_state["cam_box"]
                    if "id_box" in st.session_state:
                        del st.session_state["id_box"]
                    
                    # Pause for a moment so the user sees the success, then refresh
                    time.sleep(1.2)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")