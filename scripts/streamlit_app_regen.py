import os
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path

# -----------------------------
# Path Setup
# -----------------------------
# Add project root to sys.path so we can import from 'app'
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.face_engine.insightface_engine import get_insightface, normalize_encodings, DATA_DIR, ENCODINGS_PATH

# Define where raw images are (from your tree structure)
FACES_DIR = DATA_DIR / "raw_faces"

def generate_encodings() -> bool:
    if not FACES_DIR.exists():
        print(f"❌ Error: Faces directory not found at {FACES_DIR}")
        return False

    app = get_insightface()
    known_encodings = []
    known_ids = []

    print(f"📂 Scanning {FACES_DIR}...")

    # Iterate over student folders (e.g. 2300101419)
    for student_folder in FACES_DIR.iterdir():
        if not student_folder.is_dir():
            continue

        student_id = student_folder.name
        print(f"   Processing {student_id}...")
        
        # Iterate over images inside student folder
        valid_extensions = {".jpg", ".jpeg", ".png"}
        for img_path in student_folder.glob("*.*"):
            if img_path.suffix.lower() not in valid_extensions:
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                print(f"⚠️  No face detected in {img_path.name}")
                continue
            
            # Use largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            known_encodings.append(face.embedding)
            known_ids.append(student_id)

    if not known_encodings:
        print("❌ No faces found to encode.")
        return False

    # Save Data
    data = {
        "encodings": normalize_encodings(np.array(known_encodings)),
        "ids": known_ids
    }

    try:
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Database updated! Saved {len(known_ids)} face encodings.")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

if __name__ == "__main__":
    generate_encodings()