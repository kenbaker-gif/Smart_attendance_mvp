import os
import pickle
import sys
import cv2
import numpy as np
import requests
from supabase import create_client
from insightface.app import FaceAnalysis

# --- CONFIGURATION ---
BUCKET_NAME = "raw_faces"  # ⚠️ Ensure this bucket exists in Supabase Storage
FILE_NAME = "encodings_insightface.pkl"

# --- INIT INSIGHTFACE ---
# We initialize it here directly to avoid dependencies on your 'app' folder
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None
    # Return the embedding of the largest face
    return sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0].embedding

def main():
    print("🚀 Starting CI Encoding Generation...")
    
    # 1. Setup Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ Error: SUPABASE_URL or SUPABASE_KEY missing.")
        sys.exit(1)
        
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)

    # 2. Fetch Data
    print("📥 Fetching student list...")
    response = supabase.table("students").select("*").execute()
    students = response.data
    
    if not students:
        print("⚠️ No students found.")
        return

    known_encodings = []
    known_names = []
    
    # 3. Process Images
    print(f"🔍 Processing {len(students)} students...")
    
    for student in students:
        uid = student['id']
        name = student.get('name', 'Unknown')
        img_url = student.get('image_url')

        if not img_url:
            continue

        try:
            # Download
            resp = requests.get(img_url, timeout=10)
            if resp.status_code != 200:
                print(f"   ⚠️ Download failed for {name}")
                continue

            # Decode
            img_arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Encode
            emb = get_face_embedding(img)
            if emb is not None:
                known_encodings.append(emb)
                known_names.append(uid)
                print(f"   ✅ Encoded: {name}")
            else:
                print(f"   ⚠️ No face found: {name}")

        except Exception as e:
            print(f"   ❌ Error {name}: {e}")

    # 4. Save Locally (Temporary)
    if not known_encodings:
        print("❌ No valid encodings generated.")
        sys.exit(1)

    print("💾 Saving local pickle...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(FILE_NAME, "wb") as f:
        pickle.dump(data, f)

    # 5. Upload to Cloud
    print(f"☁️ Uploading to Supabase bucket: '{BUCKET_NAME}'...")
    try:
        with open(FILE_NAME, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=FILE_NAME,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        print("🎉 SUCCESS: Model updated in cloud!")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()