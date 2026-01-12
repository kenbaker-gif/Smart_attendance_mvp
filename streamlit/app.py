import streamlit as st
import os
import datetime
from supabase import create_client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "student_faces")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase credentials in .env file.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="Face Capture", page_icon="📸")

st.title("📸 Student Face Capture")
st.write("Direct capture to Supabase Storage.")

# Input for Student ID
student_id = st.text_input("1. Student ID", placeholder="Enter ID to create folder")

# Camera Widget
camera_image = st.camera_input("2. Take Photo")

# --- 3. UPLOAD LOGIC ---
if st.button("🚀 Save to Cloud"):
    if not student_id:
        st.warning("⚠️ Please enter a Student ID first.")
    elif not camera_image:
        st.warning("⚠️ Please take a photo using the camera above.")
    else:
        with st.spinner("Saving to Supabase..."):
            try:
                # Format: student_id/timestamp.jpg
                clean_id = student_id.strip().replace(" ", "_")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"{clean_id}/{timestamp}.jpg"

                # Upload bytes directly
                file_data = camera_image.getvalue()
                
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=file_path,
                    file=file_data,
                    file_options={"content-type": "image/jpeg"}
                )

                st.success(f"✅ Saved to folder: {clean_id}")
                st.toast(f"File {timestamp}.jpg uploaded!", icon="☁️")
                
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

# --- 4. DATA LOG ---
st.divider()
st.caption(f"Connected to: {SUPABASE_URL.split('//')[1].split('.')[0]} | Bucket: {SUPABASE_BUCKET}")