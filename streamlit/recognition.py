import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# Note: DeepFace is a heavy dependency. If you encounter errors, ensure 
# you have installed it and its required backends (e.g., TensorFlow).
try:
    from deepface import DeepFace
except ModuleNotFoundError:
    print("❌ ERROR: deepface not found. Please run: pip install deepface")
    exit()

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

# Supabase configuration (read from environment variables)
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET: Optional[str] = os.getenv("SUPABASE_BUCKET")

# File paths
RAW_FACES_DIR: Path = Path("data") / "raw_faces"
ENCODINGS_PATH: Path = Path("data") / "encodings_facenet.pkl"

# ----------------------------------------------------------------------
# SUPABASE DOWNLOAD IMPORT
# ----------------------------------------------------------------------

# This conditional import prevents the script from failing if the 
# app.utils.supabase_utils module doesn't exist.
download_all_supabase_images = None

try:
    # Assuming this module exists in your project structure
    from app.utils.supabase_utils import download_all_supabase_images
except ImportError:
    print("⚠ WARNING: Could not import supabase_utils. Supabase downloads disabled.")

# ----------------------------------------------------------------------
# LOAD DEEPFACE MODEL
# ----------------------------------------------------------------------

print("🔍 Loading DeepFace Facenet512 model...")

# The model object is built once and stored globally. DeepFace should cache this 
# internally when model_name="Facenet512" is used in DeepFace.represent().
try:
    DEEPFACE_MODEL = DeepFace.build_model("Facenet512")
    print("✅ DeepFace Facenet512 model loaded.")
except Exception as e:
    print(f"❌ ERROR: Failed to load DeepFace model: {e}")
    # Exit if the model can't be loaded, as the script is useless without it.
    exit()

# ----------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------

def _get_image_paths_for_student(student_dir: Path) -> List[Path]:
    """Return sorted list of JPG/JPEG/PNG images for a student folder."""
    return sorted(
        [p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

def _generate_face_encoding_from_image(path: Path, model: Any) -> Optional[np.ndarray]:
    """
    Generates a 512-d embedding from an image path.
    Returns None if DeepFace fails.
    
    The 'model' parameter is kept for compatibility but is intentionally
    NOT passed to DeepFace.represent() to fix the "unexpected keyword argument 'model'" error.
    """
    try:
        # NOTE: Removed 'model=model' argument to fix compatibility issue.
        # DeepFace should use the model built and cached globally (DEEPFACE_MODEL) 
        # when model_name="Facenet512" is specified.
        reps = DeepFace.represent(
            img_path=str(path),
            model_name="Facenet512",
            detector_backend="retinaface",  # Stronger face detection
            enforce_detection=False          # Continue even if face detection fails
        )
        
        # DeepFace.represent returns a list of dictionaries (one for each detected face).
        if not reps:
            print(f"⚠️ WARNING: No face detected in {path.name}.")
            return None
        
        # Extract the 512-dimensional embedding
        embedding = reps[0]["embedding"]
        return np.array(embedding, dtype=np.float32)

    except Exception as e:
        # Check for the specific type of error that indicates a face was not found
        # (Though we set enforce_detection=False, DeepFace might still raise an error 
        # depending on its internal logic or version).
        if "face could not be detected" in str(e):
             print(f"⚠️ WARNING: No face detected in {path.name} (Error: {e}).")
        else:
             print(f"❌ DeepFace failed on {path.name}: {e}")
        return None

# ----------------------------------------------------------------------
# MAIN GENERATION PIPELINE
# ----------------------------------------------------------------------

def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    """
    Generates face embeddings (encodings) from images in student folders.
    Downloads from Supabase if enabled and available.
    Saves pickle file: {"encodings": array, "ids": array}
    """
    # Ensure the input directory structure exists
    images_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # DOWNLOAD FROM SUPABASE IF ENABLED
    # -----------------------------------------------------
    if USE_SUPABASE:
        if download_all_supabase_images is None:
            print("⚠ Supabase utils missing — skipping download.")
        elif not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
            print("⚠ Supabase env vars missing — skipping download.")
        else:
            print("📦 Downloading images from Supabase...")
            # Note: Assuming download_all_supabase_images takes the required arguments
            ok = download_all_supabase_images(
                SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET,
                str(images_dir),
                clear_local=False
            )
            print("✅ Supabase download complete." if ok else "⚠️ Download failed or empty.")

    encodings: List[np.ndarray] = []
    ids: List[str] = []
    processed = 0
    skipped = 0

    # -----------------------------------------------------
    # PROCESS STUDENT FOLDERS
    # -----------------------------------------------------
    for student_dir in sorted(images_dir.iterdir()):
        if not student_dir.is_dir():
            continue

        student_id = student_dir.name
        image_paths = _get_image_paths_for_student(student_dir)

        if not image_paths:
            print(f"⚠ No images found for {student_id}, skipping.")
            continue

        print(f"📸 Processing {student_id} ({len(image_paths)} images)...")

        for img_path in image_paths:
            # Pass the globally loaded DEEPFACE_MODEL to the utility function
            enc = _generate_face_encoding_from_image(img_path, DEEPFACE_MODEL)
            if enc is None:
                skipped += 1
                continue

            encodings.append(enc)
            ids.append(student_id)
            processed += 1

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------
    if not encodings:
        print("❌ No encodings generated. Check your image folders and DeepFace logs.")
        return False

    # -----------------------------------------------------
    # SAVE PICKLE
    # -----------------------------------------------------
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "encodings": np.array(encodings, dtype=np.float32),
            "ids": np.array(ids)
        }

        with open(output_path, "wb") as fh:
            pickle.dump(data, fh)

        print(f"\n✅ Saved {len(encodings)} encodings "
              f"for {len(set(ids))} students → {output_path}")
        print(f"Summary: {processed} encoded, {skipped} skipped.")
        return True
    except Exception as e:
        print(f"❌ Failed to save encodings: {e}")
        return False

# ----------------------------------------------------------------------
# CLI ENTRYPOINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    generate_encodings()