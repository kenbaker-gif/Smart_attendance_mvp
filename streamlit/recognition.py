import os
import sys
import gc
import pickle
from pathlib import Path
from typing import List, Optional, Any
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Project root (relative, writable)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "streamlit" / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Supabase config
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

download_all_supabase_images = None
try:
    from app.utils.supabase_utils import download_all_supabase_images
except ImportError:
    print("⚠ WARNING: Could not import supabase_utils. Supabase downloads disabled.")

# -----------------------------
# InsightFace setup
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    print("❌ ERROR: insightface not found. Install: pip install insightface[onnx]")
    raise SystemExit(1)

print("🔍 Initializing InsightFace (buffalo_s, CPU)...")
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace ready.")

# -----------------------------
# Helper functions
# -----------------------------
def _get_image_paths(student_dir: Path) -> List[Path]:
    return sorted([p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")])

def _largest_face(faces) -> Optional[Any]:
    if not faces:
        return None
    def area(f): 
        x1, y1, x2, y2 = map(float, f.bbox)
        return max(0.0, (x2 - x1) * (y2 - y1))
    return max(faces, key=area)

def _generate_face_encoding_from_image(path: Path) -> Optional[np.ndarray]:
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"❌ Failed to read {path.name}")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if not faces:
            print(f"⚠️ No face detected in {path.name}")
            return None
        face = _largest_face(faces)
        if not face or getattr(face, "embedding", None) is None:
            print(f"⚠️ No usable embedding for {path.name}")
            return None
        embedding = np.array(face.embedding, dtype=np.float32)
        del img_bgr, img_rgb, faces, face
        gc.collect()
        return embedding
    except Exception as e:
        print(f"❌ InsightFace failed on {path.name}: {e}")
        gc.collect()
        return None

# -----------------------------
# Main encoding generation
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    images_dir.mkdir(parents=True, exist_ok=True)

    # Supabase download
    if USE_SUPABASE and download_all_supabase_images and SUPABASE_URL and SUPABASE_KEY and SUPABASE_BUCKET:
        temp_dir = images_dir / "_supabase_temp"
        temp_dir.mkdir(exist_ok=True)
        print("📦 Downloading images from Supabase...")
        success = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(temp_dir), clear_local=True)
        if success:
            for folder in temp_dir.iterdir():
                if folder.is_dir():
                    target = images_dir / folder.name
                    folder.rename(target)
                    print(f"✅ Added folder: {folder.name}")
            temp_dir.rmdir()
        else:
            print("⚠️ Supabase download failed or empty.")

    encodings, ids = [], []
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
    print(f"📁 Found {len(student_dirs)} student folders to process.")

    processed, skipped = 0, 0
    for student_dir in student_dirs:
        student_id = student_dir.name
        image_paths = _get_image_paths(student_dir)
        if not image_paths:
            print(f"⚠ No images for {student_id}, skipping.")
            continue
        print(f"📸 Processing {student_id} ({len(image_paths)} images)...")
        for img_path in image_paths:
            emb = _generate_face_encoding_from_image(img_path)
            if emb is None:
                skipped += 1
                continue
            encodings.append(emb)
            ids.append(student_id)
            processed += 1

    if not encodings:
        print("❌ No encodings generated.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump({"encodings": np.array(encodings, dtype=np.float32), "ids": np.array(ids)}, fh)

    print(f"\n✅ Saved {len(encodings)} encodings for {len(set(ids))} students → {output_path}")
    print(f"Summary: {processed} encoded, {skipped} skipped.")
    return True

# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    generate_encodings()
