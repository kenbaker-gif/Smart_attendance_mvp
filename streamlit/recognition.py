import traceback
from typing import List, Optional, Dict, Any
import os
import sys
import pickle
import gc
from pathlib import Path
import numpy as np
import cv2

# ----------------------------------------------------------------------
# PROJECT ROOT SETUP
# ----------------------------------------------------------------------
ABSOLUTE_PROJECT_ROOT = "/home/kenbaker-gif/smart_attendance_system"
if sys.path[0] != ABSOLUTE_PROJECT_ROOT:
    sys.path.insert(0, ABSOLUTE_PROJECT_ROOT)

# ----------------------------------------------------------------------
# SUPABASE CONFIGURATION
# ----------------------------------------------------------------------
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET: Optional[str] = os.getenv("SUPABASE_BUCKET")

# ----------------------------------------------------------------------
# FORCE SUPABASE DOWNLOAD IMPORT
# ----------------------------------------------------------------------
download_all_supabase_images = None
try:
    from app.utils.supabase_utils import download_all_supabase_images
except ImportError:
    print("⚠ WARNING: Could not import supabase_utils. Supabase downloads disabled.")

# ----------------------------------------------------------------------
# THREAD/MEMORY TUNING FOR LOW-RAM ENVIRONMENTS
# ----------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ----------------------------------------------------------------------
# INSIGHTFACE SETUP
# ----------------------------------------------------------------------
try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    print("❌ ERROR: insightface not found. Install: pip install insightface[onnx]")
    raise SystemExit(1)

print("🔍 Initializing InsightFace (buffalo_l, CPU)...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace (buffalo_l) ready.")

# ----------------------------------------------------------------------
# PATH CONFIG
# ----------------------------------------------------------------------
RAW_FACES_DIR = Path(ABSOLUTE_PROJECT_ROOT) / "streamlit" / "data" / "raw_faces"
ENCODINGS_PATH = Path(ABSOLUTE_PROJECT_ROOT) / "streamlit" / "data" / "encodings_insightface.pkl"

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def _get_image_paths_for_student(student_dir: Path) -> List[Path]:
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
        if face is None or getattr(face, "embedding", None) is None:
            print(f"⚠️ No usable embedding for {path.name}")
            return None
        embedding = np.array(face.embedding, dtype=np.float32)
        del img_bgr, img_rgb, faces, face
        gc.collect()
        return embedding
    except Exception as e:
        print(f"❌ InsightFace failed on {path.name}: {e}")
        try: del img_bgr, img_rgb, faces
        except NameError: pass
        gc.collect()
        return None

# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    images_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # DOWNLOAD MISSING STUDENTS FROM SUPABASE
    # -----------------------------------------------------
    if USE_SUPABASE and download_all_supabase_images and SUPABASE_URL and SUPABASE_KEY and SUPABASE_BUCKET:
        # List existing student folders
        existing_students = {p.name for p in images_dir.iterdir() if p.is_dir()}
        print(f"📂 Existing student folders: {len(existing_students)}")

        # Download all images to a temporary directory
        temp_dir = images_dir / "_supabase_temp"
        temp_dir.mkdir(exist_ok=True)
        print("📦 Downloading images from Supabase (temporary)...")
        ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(temp_dir), clear_local=True)
        if ok:
            # Move only new student folders
            for student_folder in temp_dir.iterdir():
                if student_folder.is_dir() and student_folder.name not in existing_students:
                    target_folder = images_dir / student_folder.name
                    student_folder.rename(target_folder)
                    print(f"✅ Added new student folder: {student_folder.name}")
            # Cleanup temp folder
            temp_dir.rmdir()
        else:
            print("⚠️ Supabase download failed or returned no files.")
    elif USE_SUPABASE:
        print("⚠ Supabase disabled or missing utils/env vars — skipping download.")

    # -----------------------------------------------------
    # GENERATE ENCODINGS
    # -----------------------------------------------------
    encodings: List[np.ndarray] = []
    ids: List[str] = []
    processed, skipped = 0, 0

    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
    print(f"📁 Found {len(student_dirs)} student folders to process.")

    for student_dir in student_dirs:
        student_id = student_dir.name
        image_paths = _get_image_paths_for_student(student_dir)
        if not image_paths:
            print(f"⚠ No images for {student_id}, skipping.")
            continue
        print(f"📸 Processing {student_id} ({len(image_paths)} images)...")
        for img_path in image_paths:
            enc = _generate_face_encoding_from_image(img_path)
            if enc is None:
                skipped += 1
                continue
            encodings.append(enc)
            ids.append(student_id)
            processed += 1

    if not encodings:
        print("❌ No encodings generated.")
        return False

    # Save pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump({"encodings": np.array(encodings, dtype=np.float32), "ids": np.array(ids)}, fh)
    print(f"\n✅ Saved {len(encodings)} encodings for {len(set(ids))} students → {output_path}")
    print(f"Summary: {processed} encoded, {skipped} skipped.")
    return True

# ----------------------------------------------------------------------
# CLI ENTRYPOINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    generate_encodings()
