import shutil
import os
from pathlib import Path
from typing import List, Union
from supabase import create_client
from dotenv import load_dotenv

# -----------------------------
# Helpers
# -----------------------------
def _normalize_list_response(resp) -> List[dict]:
    """Ensures we get a list of file/folder objects regardless of SDK version."""
    if resp is None: return []
    if isinstance(resp, list): return resp
    if isinstance(resp, dict):
        for key in ("data", "files", "list"):
            if isinstance(resp.get(key), list):
                return resp[key]
    return []

def _download_bytes_from_response(res) -> Union[bytes, None]:
    """Extracts raw bytes from the Supabase download response."""
    if isinstance(res, (bytes, bytearray)): return bytes(res)
    if isinstance(res, dict):
        data = res.get("data") or res.get("body") or res.get("content")
        if data: return bytes(data) if isinstance(data, (bytes, bytearray)) else data.encode()
    return None

# -----------------------------
# 1. Download Images (Sync Down)
# -----------------------------
def download_all_supabase_images(
    supabase_url: str,
    supabase_key: str,
    supabase_bucket: str,
    local_images_dir: str,
    clear_local: bool = True,
) -> bool:
    """
    Production version of the recursive downloader.
    Maps: Supabase Bucket/StudentID/1.jpg -> local_dir/StudentID/1.jpg
    """
    local_path = Path(local_images_dir)
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        storage_api = supabase.storage.from_(supabase_bucket)
    except Exception as e:
        print(f"❌ Supabase Client Error: {e}")
        return False

    # Prepare local directory
    if clear_local and local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    download_count = 0

    try:
        # Step 1: List root to find student folders
        print(f"📂 Scanning bucket root: {supabase_bucket}")
        root_items = _normalize_list_response(storage_api.list("", options={"limit": 1000}))
        
        folder_names = [item['name'] for item in root_items if not item['name'].startswith('.')]

        for student_id in folder_names:
            # Step 2: List contents of each folder
            sub_items = _normalize_list_response(storage_api.list(student_id))
            
            for file_entry in sub_items:
                file_name = file_entry.get("name")
                
                if file_name and Path(file_name).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    remote_path = f"{student_id}/{file_name}"
                    
                    local_student_dir = local_path / student_id
                    local_student_dir.mkdir(parents=True, exist_ok=True)
                    local_file_path = local_student_dir / file_name

                    # Step 3: Download
                    res = storage_api.download(remote_path)
                    data = _download_bytes_from_response(res)
                    
                    if data:
                        with open(local_file_path, "wb") as f:
                            f.write(data)
                        download_count += 1
                        print(f"   ✅ Saved: {student_id}/{file_name}")

        print(f"✨ Successfully downloaded {download_count} images.")
        return download_count > 0

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

# -----------------------------
# 2. Upload Encodings (Sync Up)
# -----------------------------
def upload_encodings_to_supabase(
    supabase_url: str,
    supabase_key: str,
    supabase_bucket: str,
    local_file_path: str,
    remote_file_name: str = "encodings_insightface.pkl"
) -> bool:
    """
    Uploads the generated pickle file to Supabase Storage.
    """
    path = Path(local_file_path)
    if not path.exists():
        print(f"❌ Upload failed: File not found at {path}")
        return False

    try:
        supabase = create_client(supabase_url, supabase_key)
        
        # ✅ FIX 1: Ensure upload goes to the right folder if needed
        # (Though usually the generator script handles the upload, this is a good backup)
        target_path = f"encodings/{remote_file_name}" 
        
        print(f"☁️ Uploading {target_path} to {supabase_bucket}...")
        
        with open(path, "rb") as f:
            supabase.storage.from_(supabase_bucket).upload(
                path=target_path,
                file=f,
                file_options={"upsert": "true"}
            )
            
        print("✅ Upload Successful!")
        return True
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        return False

# -----------------------------
# 3. Download Encodings (Startup)
# -----------------------------
def download_encodings_from_supabase(local_save_path: str, bucket_name: str = None) -> bool:
    """
    Downloads the pickle file directly from 'raw_faces/encodings/...'
    """
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    # ✅ FIX 2: Force bucket to be 'raw_faces' (The one we know works)
    bucket = "raw_faces"
    
    # ✅ FIX 3: Point to the correct SUBFOLDER
    source_path = "encodings/encodings_insightface.pkl"
    
    if not url or not key:
        print("Supabase credentials missing.")
        return False
        
    try:
        sb = create_client(url, key)
        print(f"⬇️ Downloading {source_path} from {bucket}...")
        
        # ✅ FIX 4: Download the specific file path
        res = sb.storage.from_(bucket).download(source_path)
        
        data = _download_bytes_from_response(res)
        
        if data:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
            
            with open(local_save_path, 'wb') as f:
                f.write(data)
            print("✅ Encodings downloaded successfully.")
            return True
        else:
            print("❌ Downloaded file was empty.")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download encodings: {e}")
        return False