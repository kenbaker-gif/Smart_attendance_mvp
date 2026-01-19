import os
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------
# Path & Env Setup
# -----------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load Secrets
load_dotenv(PROJECT_ROOT / "secrets.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "biometric-data")

# Import Engine & Utils
from app.face_engine.insightface_engine import get_insightface, normalize_encodings, DATA_DIR, ENCODINGS_PATH
# Import YOUR specific utils file
from app.utils.supabase_utils import download_all_supabase_images, upload_encodings_to_supabase

# Local Faces Directory
FACES_DIR = DATA_DIR / "raw_faces"

def generate_encodings() -> bool:
    # -----------------------------------------
    # 1. CLOUD SYNC: Download latest images
    # -----------------------------------------
    print("--- 1. Syncing from Supabase ---")
    if SUPABASE_URL and SUPABASE_KEY:
        success = download_all_supabase_images(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY,
            supabase_bucket=SUPABASE_BUCKET,
            local_images_dir=str(FACES_DIR),
            clear_local=True # Wipe old local files to ensure exact mirror
        )
        if not success:
            print("⚠️ Download warning: Proceeding with existing local files (if any).")
    else:
        print("⚠️ No Supabase credentials found. Skipping download.")

    # -----------------------------------------
    # 2. LOCAL PROCESSING: Generate Encodings
    # -----------------------------------------
    print("\n--- 2. Generating Encodings ---")
    if not FACES_DIR.exists():
        print(f"❌ Error: Faces directory not found at {FACES_DIR}")
        return False

    app = get_insightface()
    known_encodings = []
    known_ids = []

    print(f"📂 Scanning {FACES_DIR}...")
    
    found_faces = False
    
    # Iterate over student folders
    for student_folder in FACES_DIR.iterdir():
        if not student_folder.is_dir(): continue

        student_id = student_folder.name
        
        for img_path in student_folder.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]: continue
            
            img = cv2.imread(str(img_path))
            if img is None: continue

            faces = app.get(img)
            if not faces: 
                # Optional: print(f"Skipping {img_path.name} - no face")
                continue
            
            # Use largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            known_encodings.append(face.embedding)
            known_ids.append(student_id)
            found_faces = True

    if not found_faces:
        print("❌ No faces found to encode.")
        return False

    # Save Pickle Locally
    data = {
        "encodings": normalize_encodings(np.array(known_encodings)),
        "ids": known_ids
    }
    
    try:
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Local Database updated! ({len(known_ids)} faces)")
    except Exception as e:
        print(f"❌ Error saving local file: {e}")
        return False

    # -----------------------------------------
    # 3. CLOUD SYNC: Upload result
    # -----------------------------------------
    print("\n--- 3. Syncing to Cloud ---")
    if SUPABASE_URL and SUPABASE_KEY:
        upload_encodings_to_supabase(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY,
            supabase_bucket=SUPABASE_BUCKET,
            local_file_path=str(ENCODINGS_PATH)
        )
    else:
        print("⚠️ No Supabase credentials. Skipping upload.")
    
    return True

if __name__ == "__main__":
    generate_encodings()