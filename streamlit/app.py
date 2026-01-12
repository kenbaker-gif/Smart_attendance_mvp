import streamlit as st
import os
from supabase import create_client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "student_faces")

if URL and not URL.endswith("/"):
    URL += "/"

if not URL or not KEY:
    st.error("❌ Missing credentials in .env")
    st.stop()

supabase = create_client(URL, KEY)

# --- 2. UI ---
st.set_page_config(page_title="Sequential Capture", page_icon="📸")
st.title("📸 Student Face Capture")

student_id = st.text_input("Enter Student ID").strip()
camera_image = st.camera_input("Position face in center")

# --- 3. UPLOAD LOGIC ---
if st.button("🚀 Save to Supabase"):
    if not student_id or not camera_image:
        st.warning("⚠️ ID and Photo required!")
    else:
        try:
            # A. Get list of existing files
            folder_path = student_id
            res = supabase.storage.from_(BUCKET).list(folder_path)
            
            # Count JPGs
            existing_files = [f['name'] for f in res if f['name'].lower().endswith('.jpg')]
            next_num = len(existing_files) + 1
            
            # B. COLLISION CHECK LOOP
            # This prevents the '409 Duplicate' error by checking if the name is taken
            uploaded = False
            while not uploaded:
                file_name = f"{next_num}.jpg"
                full_path = f"{folder_path}/{file_name}"
                
                # Try to upload
                try:
                    img_bytes = camera_image.getvalue()
                    supabase.storage.from_(BUCKET).upload(
                        path=full_path,
                        file=img_bytes,
                        file_options={"content-type": "image/jpeg"}
                    )
                    st.success(f"✅ Saved as {full_path}")
                    uploaded = True # Exit loop
                except Exception as upload_err:
                    # If duplicate error occurs, increment and try again
                    if "already exists" in str(upload_err).lower() or "409" in str(upload_err):
                        next_num += 1
                    else:
                        raise upload_err # If it's a different error, stop and report it
            
        except Exception as e:
            st.error(f"❌ Storage Error: {e}")

st.divider()