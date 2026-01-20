import os
import pickle
import sys
import cv2
import numpy as np
from supabase import create_client
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

# --- CONFIGURATION ---
BUCKET_NAME = "raw_faces"
OUTPUT_FILE = "encodings_insightface.pkl"
UPLOAD_PATH = "encodings/" + OUTPUT_FILE 

# --- INIT ---
load_dotenv("secrets.env")

# Initialize InsightFace (Heavy model, so we do it globally)
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None
    return sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0].embedding

# ✅ UPDATED: Now accepts 'progress_callback'
def generate_encodings(progress_callback=None):
    print(f"🚀 Starting Generation in bucket: '{BUCKET_NAME}'...", flush=True)
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ Error: Supabase credentials missing.", flush=True)
        return False
        
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"❌ Connection Failed: {e}", flush=True)
        return False

    # 1. Get list of Folders
    print(f"📂 Scanning root of '{BUCKET_NAME}'...", flush=True)
    try:
        root_items = supabase.storage.from_(BUCKET_NAME).list()
    except Exception as e:
        print(f"❌ Error accessing bucket: {e}", flush=True)
        return False

    known_encodings = []
    known_names = []
    
    # Filter for valid student folders only
    valid_folders = [item for item in root_items 
                     if not item['name'].startswith('.') 
                     and item['name'] != 'encodings' 
                     and '.' not in item['name']]
    
    total_students = len(valid_folders)
    print(f"🔍 Found {total_students} student folders.", flush=True)

    # 2. Loop through each folder
    for i, item in enumerate(valid_folders):
        
        # ✅ UPDATE STREAMLIT PROGRESS BAR (If provided)
        if progress_callback:
            # Calculate percentage (0.0 to 1.0)
            percent = (i + 1) / total_students
            # Update the bar with a message
            progress_callback(percent, f"Processing {item['name']}...")

        student_id = item['name']
        print(f"   📂 Checking Student: {student_id}...", end="", flush=True)

        try:
            student_files = supabase.storage.from_(BUCKET_NAME).list(student_id)
            
            found_image = False
            for file in student_files:
                file_name = file['name']
                if file_name.startswith('.'): continue
                
                full_path = f"{student_id}/{file_name}"
                
                try:
                    file_data = supabase.storage.from_(BUCKET_NAME).download(full_path)
                    img_arr = np.frombuffer(file_data, np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if img is None: continue

                    emb = get_face_embedding(img)
                    if emb is not None:
                        known_encodings.append(emb)
                        known_names.append(student_id)
                        print(f" ✅ Encoded", flush=True)
                        found_image = True
                        break 
                except Exception:
                    continue

            if not found_image:
                print(f" ⚠️ No valid face found", flush=True)

        except Exception as e:
            print(f" ❌ Error accessing folder: {e}", flush=True)

    # 3. Save & Upload
    if not known_encodings:
        print("❌ No valid encodings generated.", flush=True)
        return False

    print("💾 Saving pickle file locally...", flush=True)
    data = {"encodings": known_encodings, "names": known_names}
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"☁️ Uploading to '{BUCKET_NAME}/{UPLOAD_PATH}'...", flush=True)
    try:
        with open(OUTPUT_FILE, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=UPLOAD_PATH,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        print("🎉 SUCCESS: Encodings updated!", flush=True)
        return True
    except Exception as e:
        print(f"❌ Upload Failed: {e}", flush=True)
        return False

if __name__ == "__main__":
    generate_encodings()