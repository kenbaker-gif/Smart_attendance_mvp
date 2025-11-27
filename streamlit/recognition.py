import os
import pickle
from pathlib import Path
import face_recognition
import numpy as np # Added for robust type hinting (though not strictly required here)
from typing import Tuple, List, Optional

# Local import (ensure your project root is on sys.path when running standalone)
try:
    # Assuming this import structure is correct relative to the project root
    from app.utils.supabase_utils import download_all_supabase_images
except ImportError:
    # Use a more specific exception for clarity
    download_all_supabase_images = None
    print("WARNING: Could not import app.utils.supabase_utils. Supabase download functionality is disabled.")


# CONFIG
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET: Optional[str] = os.getenv("SUPABASE_BUCKET")

RAW_FACES_DIR: Path = Path("data") / "raw_faces"
ENCODINGS_PATH: Path = Path("data") / "encodings_facenet.pkl"


def _get_image_paths_for_student(student_dir: Path) -> List[Path]:
    """Return list of image files for a student folder (jpg/jpeg/png)."""
    # Use a generator expression for efficiency
    return sorted([
        p for p in student_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])


def _generate_face_encoding_from_image(path: Path) -> Optional[np.ndarray]:
    """Return first face encoding (as a numpy array) from image or None if not found/error."""
    try:
        # Load image file
        img = face_recognition.load_image_file(str(path))
        
        # Locate faces using HOG model
        locs = face_recognition.face_locations(img, model="hog")
        
        # Check for exactly one face for high-quality training data
        if len(locs) != 1:
            return None
            
        # Generate encoding
        encs = face_recognition.face_encodings(img, locs)
        
        if encs:
            # Return the encoding as a numpy array
            return encs[0]
        return None
    except Exception as e:
        # Use simple print since this is a utility/cli script
        print(f"❌ Error processing {path.name}: {e}")
        return None


def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    """
    Ensure images are present (download from Supabase if configured), then
    iterate student folders and build encodings pickle.
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    images_dir.mkdir(parents=True, exist_ok=True)

    # If using Supabase, download into nested structure
    if USE_SUPABASE:
        if download_all_supabase_images is None:
            print("❌ Supabase utils not available. Set up app.utils.supabase_utils.")
            # Continue running with local files instead of failing the function
        elif not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
            print("❌ Supabase environment variables are missing. Skipping download.")
        else:
            print("📦 USE_SUPABASE=True — downloading images from Supabase into local folders...")
            
            # Assuming download_all_supabase_images signature: (url, key, bucket, local_base_path, clear_local)
            ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=False)
            
            if not ok:
                print("⚠ Supabase download failed or returned 0 files — attempting to continue with local files.")
            else:
                print("✅ Supabase download complete.")


    encodings: List[np.ndarray] = []
    ids: List[str] = []
    processed_files = 0
    skipped_files = 0

    # Iterate student directories
    for student_dir in sorted(images_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_id = student_dir.name
        image_paths = _get_image_paths_for_student(student_dir)
        
        if not image_paths:
            print(f"⚠ No images found for {student_id}, skipping.")
            continue

        print(f"📸 Processing student {student_id} ({len(image_paths)} images)...")
        for img_path in image_paths:
            enc = _generate_face_encoding_from_image(img_path)
            
            if enc is None:
                skipped_files += 1
                # Removed detailed print for every skipped file to reduce verbosity
                continue
                
            encodings.append(enc)
            ids.append(student_id)
            processed_files += 1
            
            # Print feedback only for successfully encoded files (less noisy output)
            # print(f"  ✅ Encoded {img_path.name}") 

    if not encodings:
        print("❌ No encodings generated. Check image folders and visibility of faces.")
        return False

    # Save to pickle
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Combine encodings/ids into a single dictionary
        data = {"encodings": np.array(encodings), "ids": np.array(ids)}
        
        with open(output_path, "wb") as fh:
            pickle.dump(data, fh)
            
        unique_students = len(set(ids))
        print(f"\n✅ Saved {len(encodings)} encodings for {unique_students} students to {output_path}")
        print(f"Summary: {processed_files} faces encoded, {skipped_files} files skipped.")
        return True
    except Exception as e:
        print(f"❌ Failed to save encodings to {output_path}: {e}")
        return False


if __name__ == "__main__":
    generate_encodings()