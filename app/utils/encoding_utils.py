import os
import shutil
from pathlib import Path
import face_recognition
import pickle
from typing import List, Union

# -----------------------------
# SUPABASE CONFIG & INITIALIZATION
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase = None
if USE_SUPABASE:
    try:
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
            print("⚠ Supabase enabled but one or more SUPABASE_* env vars are missing.")
            supabase = None
        else:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✅ Supabase client initialized.")
    except Exception as e:
        print(f"⚠ Failed to initialize Supabase client: {e}")
        supabase = None


# -----------------------------
# HELPER FUNCTIONS FOR SUPABASE
# -----------------------------
def _normalize_list_response(resp) -> List[dict]:
    """
    Normalizes different possible list() return shapes from Supabase client
    into a list of dicts containing at least a 'name' or 'id' key.
    """
    if resp is None:
        return []
    if isinstance(resp, dict):
        for key in ("data", "files", "list"):
            if key in resp and isinstance(resp[key], list):
                return resp[key]
        if "error" in resp:
            return []
    if isinstance(resp, list):
        return resp
    return []


def _download_bytes_from_response(res) -> Union[bytes, None]:
    """
    Convert various SDK download() responses to raw bytes.
    Returns bytes on success, None on failure.
    """
    if res is None:
        return None
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if isinstance(res, dict):
        if res.get("error"):
            return None
        for key in ("data", "body", "content"):
            val = res.get(key)
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
            if isinstance(val, str):
                return val.encode()
        return None
    try:
        if hasattr(res, "read"):
            return res.read()
    except Exception:
        pass
    try:
        return bytes(res)
    except Exception:
        return None


# -----------------------------------------------------------
# DOWNLOAD IMAGES FROM SUPABASE STORAGE (FIXED FOR FLAT PATHS)
# -----------------------------------------------------------
def download_all_supabase_images(local_images_dir: str) -> bool:
    """
    Downloads all images from the configured Supabase bucket, forcing the 
    creation of the required nested structure by extracting the student ID
    from the beginning of the filename.
    """
    if supabase is None or SUPABASE_BUCKET is None:
        print("❌ Supabase client not initialized or bucket name missing.")
        return False
        
    local_images_path = Path(local_images_dir)
    storage_api = supabase.storage.from_(SUPABASE_BUCKET)
    
    print(f"📦 Starting download from Supabase bucket: {SUPABASE_BUCKET}")
    
    # 1. Clear the local directory to ensure fresh data
    try:
        if local_images_path.exists():
            shutil.rmtree(local_images_path)
        local_images_path.mkdir(parents=True, exist_ok=True)
        print(f"🧹 Cleared and created local directory: {local_images_dir}")
    except Exception as e:
        print(f"❌ Failed to clear/create local directory: {e}")
        return False

    # 2. List ALL files recursively in the bucket
    try:
        all_files_raw = storage_api.list("", options={"limit": 1000, "deep": True})
        all_files = _normalize_list_response(all_files_raw)
    except Exception as e:
        print(f"❌ Failed to list files from Supabase: {e}")
        return False

    download_count = 0
    
    if not all_files:
        print("⚠️ Supabase bucket list returned no files.")
        return False
        
    # 3. Download and save each file
    for file_entry in all_files:
        remote_path = file_entry.get('id') or file_entry.get('name')
        
        if not remote_path or remote_path.endswith('/'): # Skip directories
            continue
            
        # Get just the filename (e.g., 2400102415_face.jpg)
        filename = Path(remote_path).name 
        
        # --- CRITICAL FIX: Manually extract student ID from the filename ---
        try:
            student_id = filename[:10]
            # Ensure the extracted ID is the correct length and numeric
            if not (len(student_id) == 10 and student_id.isdigit()):
                 print(f"⚠️ Skipping file, extracted ID is invalid: {filename}")
                 continue
        except IndexError:
            print(f"⚠️ Skipping file with short name (less than 10 chars): {filename}")
            continue

        # Check for valid image extensions
        if Path(filename).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # Construct the local path using the student_id as a folder
        # This FORCES: data/raw_faces/2400102415/filename.jpg
        local_file_path = local_images_path / student_id / filename
        
        print(f"DEBUG (DOWNLOAD FIX): Remote Path: {remote_path} -> Local Path: {local_file_path}") 

        local_file_path.parent.mkdir(parents=True, exist_ok=True) # Creates the student_id folder

        try:
            # Download the file content
            file_data_raw = storage_api.download(remote_path)
            file_data = _download_bytes_from_response(file_data_raw)
            
            if file_data is not None and file_data:
                with open(local_file_path, "wb") as f:
                    f.write(file_data)
                download_count += 1
            else:
                print(f"❌ Failed to download/convert file data for: {remote_path}. Content was empty.")

        except Exception as e:
            print(f"❌ Error during download or save for {remote_path}: {e}")

    print(f"✅ Download complete. Saved {download_count} files to {local_images_dir}.")
    
    if download_count == 0 and len(all_files) > 0:
         print("⚠️ Files listed but none saved.")
         return False

    return True


# -----------------------------------------------------------
# GENERATE FACE ENCODINGS (With Final Debug Check)
# -----------------------------------------------------------
def generate_encodings(images_dir: str = "data/raw_faces", output_path: str = "data/encodings_facenet.pkl") -> bool:
    """
    Downloads (if configured) and reads images from images_dir, generates encodings,
    and saves a pickle { 'encodings': [...], 'ids': [...] } to output_path.

    Returns True on success, False otherwise.
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    # 1. Download images if Supabase is enabled
    images_dir.mkdir(parents=True, exist_ok=True) 
    
    if USE_SUPABASE and supabase is not None:
        ok = download_all_supabase_images(str(images_dir))
        if not ok:
            print("⚠ Supabase download step reported failure — continuing with any local files present.")

    encodings = []
    ids = []

    # 2. Iterate and Encode local images
    for student_folder in sorted(images_dir.iterdir()):
        if not student_folder.is_dir():
            continue

        student_id = student_folder.name
        
        # ADDED DEBUG LINE: CONFIRMS FOLDER READ SUCCESS
        print(f"DEBUG ENCODER: Found folder/ID: {student_id}") 
        
        # Check for valid image files
        image_files = sorted([p for p in student_folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

        if not image_files:
            print(f"⚠️  No images found for {student_id}")
            continue

        print(f"📸 Processing student {student_id} ({len(image_files)} images)...")

        for img_path in image_files:
            try:
                image = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(image, model="hog")

                if not face_locations:
                    print(f"  ⚠️  No face detected in {img_path.name}")
                    continue

                face_encs = face_recognition.face_encodings(image, face_locations)

                if face_encs:
                    encodings.append(face_encs[0])
                    ids.append(student_id)
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")

    if not encodings:
        print("\n❌ No encodings generated. Check your image folder structure or Supabase files.")
        return False

    # 3. Save encodings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "wb") as f:
            pickle.dump({"encodings": encodings, "ids": ids}, f)
        print(f"\n✅ Saved {len(encodings)} encodings to {output_path}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to save encodings: {e}")
        return False