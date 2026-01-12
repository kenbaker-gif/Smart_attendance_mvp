import streamlit as st
import os
from supabase import create_client
from dotenv import load_dotenv

# --- 1. CONFIGURATION & CREDENTIALS ---
load_dotenv()

# Get variables from .env
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "student_faces")

# Fix trailing slash error automatically
if URL and not URL.endswith("/"):
    URL += "/"

# Stop app if credentials are missing
if not URL or not KEY:
    st.error("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env file.")
    st.stop()

# Initialize Supabase
try:
    supabase = create_client(URL, KEY)
except Exception as e:
    st.error(f"Failed to connect to Supabase: {e}")
    st.stop()

# --- 2. UI SETUP ---
st.set_page_config(page_title="Sequential Capture", page_icon="📸")

st.title("📸 Student Face Capture")
st.write("Images are stored as 1.jpg, 2.jpg, etc., inside the student's folder.")

# Input Field
student_id = st.text_input("Enter Student ID (e.g., 24001)").strip()

# Camera Widget
camera_image = st.camera_input("Position face in the center")

# --- 3. LOGIC: COUNT & UPLOAD ---
if st.button("🚀 Save to Supabase"):
    if not student_id:
        st.warning("⚠️ Please enter a Student ID first.")
    elif not camera_image:
        st.warning("⚠️ Please take a photo first.")
    else:
        try:
            # A. Check existing files to determine next number
            # We list the folder named after the student_id
            folder_path = student_id
            
            # Use Supabase storage list to find files in that folder
            res = supabase.storage.from_(BUCKET).list(folder_path)
            
            # Filter list to only count actual .jpg images (ignoring system files)
            existing_files = [f for f in res if f['name'].lower().endswith('.jpg')]
            
            # B. Determine next filename
            next_number = len(existing_files) + 1
            file_name = f"{next_number}.jpg"
            full_storage_path = f"{folder_path}/{file_name}"
            
            # C. Upload to Supabase
            with st.spinner(f"Saving as {file_name}..."):
                img_bytes = camera_image.getvalue()
                
                supabase.storage.from_(BUCKET).upload(
                    path=full_storage_path,
                    file=img_bytes,
                    file_options={"content-type": "image/jpeg"}
                )
                
            st.success(f"✅ Success! Saved to `{full_storage_path}`")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Storage Error: {e}")

# --- 4. FOOTER ---
st.divider()
st.caption(f"Bucket: `{BUCKET}` | Logic: Sequential (n+1)")