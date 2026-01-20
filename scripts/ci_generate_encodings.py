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

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None
    return sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0].embedding

def main():
    print(f"🚀 Starting Recursive Generation in bucket: '{BUCKET_NAME}'...", flush=True)
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ Error: Supabase credentials missing.", flush=True)
        sys.exit(1)
        
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"❌ Connection Failed: {e}", flush=True)
        sys.exit(1)

    # 1. Get list of Folders (Student IDs)
    print(f"📂 Scanning root of '{BUCKET_NAME}'...", flush=True)
    try:
        root_items = supabase.storage.from_(BUCKET_NAME).list()
    except Exception as e:
        print(f"❌ Error accessing bucket: {e}", flush=True)
        return

    known_encodings = []
    known_names = []
    
    # 2. Loop through each folder
    for item in root_items:
        folder_name = item['name']
        
        # Skip the 'encodings' folder and any loose files in root
        if folder_name == 'encodings' or folder_name.startswith('.'):
            continue
            
        # Heuristic: If it has an extension (like .pkl or .jpg), it's a file, not a student folder. Skip it.
        if '.' in folder_name:
            continue

        student_id = folder_name
        print(f"   📂 Checking Student: {student_id}...", end="", flush=True)

        try:
            # List files INSIDE this student's folder
            student_files = supabase.storage.from_(BUCKET_NAME).list(folder_name)
            
            # Find the first valid image file
            found_image = False
            for file in student_files:
                file_name = file['name']
                if file_name.startswith('.'): continue
                
                # Construct path: "2400102415/1.jpg"
                full_path = f"{folder_name}/{file_name}"
                
                # Download
                try:
                    file_data = supabase.storage.from_(BUCKET_NAME).download(full_path)
                    img_arr = np.frombuffer(file_data, np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if img is None: continue

                    emb = get_face_embedding(img)
                    if emb is not None:
                        known_encodings.append(emb)
                        known_names.append(student_id) # Use the FOLDER NAME as ID
                        print(f" ✅ Encoded", flush=True)
                        found_image = True
                        break # Stop after finding one valid face for this student
                except Exception as inner_e:
                    continue

            if not found_image:
                print(f" ⚠️ No valid face found in folder", flush=True)

        except Exception as e:
            print(f" ❌ Error accessing folder: {e}", flush=True)

    # 3. Save & Upload
    if not known_encodings:
        print("❌ No valid encodings generated.", flush=True)
        sys.exit(1)

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
    except Exception as e:
        print(f"❌ Upload Failed: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()