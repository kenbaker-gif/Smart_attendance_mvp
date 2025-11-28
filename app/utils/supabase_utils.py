import os
import shutil
from pathlib import Path
from typing import List, Union
from supabase import create_client

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
# DOWNLOAD IMAGES FROM SUPABASE STORAGE
# -----------------------------------------------------------
def download_all_supabase_images(
    supabase_url: str,
    supabase_key: str,
    supabase_bucket: str,
    local_images_dir: str,
    clear_local: bool = True,
) -> bool:
    """
    Downloads all images from the configured Supabase bucket, forcing the
    creation of the required nested structure by extracting the student ID
    from the beginning of the filename.

    Initializes the Supabase client using the provided URL and Key.
    """

    # Initialize Supabase client with provided credentials
    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client with provided URL/Key: {e}")
        return False

    local_images_path = Path(local_images_dir)
    storage_api = supabase.storage.from_(supabase_bucket)

    print(f"📦 Starting download from Supabase bucket: {supabase_bucket}")

    # 1. Clear the local directory (controlled by clear_local flag)
    try:
        if clear_local and local_images_path.exists():
            print(f"🧹 Clearing existing local directory: {local_images_dir}")
            shutil.rmtree(local_images_path)

        local_images_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Local directory ensured: {local_images_dir}")
    except Exception as e:
        print(f"❌ Failed to manage local directory: {e}")
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
        return True # Considered successful if nothing to download

    # 3. Download and save each file
    for file_entry in all_files:
        remote_path = file_entry.get('id') or file_entry.get('name')

        if not remote_path or remote_path.endswith('/'): # Skip directories
            continue

        # Get just the filename (e.g., 2400102415_face.jpg)
        filename = Path(remote_path).name

        # --- CRITICAL: Manually extract student ID from the filename ---
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
        local_file_path = local_images_path / student_id / filename

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

    return download_count > 0