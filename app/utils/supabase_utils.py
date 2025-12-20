import shutil
from pathlib import Path
from typing import List, Union
from supabase import create_client

def _normalize_list_response(resp) -> List[dict]:
    """Ensures we get a list of file/folder objects regardless of SDK version."""
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
    """Extracts raw bytes from the Supabase download response."""
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
    try:
        if hasattr(res, "read"):
            return res.read()
    except Exception:
        pass
    return None

def download_all_supabase_images(
    supabase_url: str,
    supabase_key: str,
    supabase_bucket: str,
    local_images_dir: str,
    clear_local: bool = True,
) -> bool:
    """
    Recursive downloader: 
    1. Lists root to find Student ID folders.
    2. Lists inside each folder to find images (e.g., 1.jpg).
    """
    local_path = Path(local_images_dir)

    try:
        supabase = create_client(supabase_url, supabase_key)
        storage_api = supabase.storage.from_(supabase_bucket)
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        return False

    # Prepare local directory
    try:
        if clear_local and local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to prepare local directory: {e}")
        return False

    download_count = 0

    try:
        # Step 1: List the root of the bucket to find Student ID folders
        print(f"📂 Scanning bucket: {supabase_bucket} for student folders...")
        root_items = _normalize_list_response(storage_api.list("", options={"limit": 1000}))
        
        if not root_items:
            print("⚠ No folders or files found in the bucket root.")
            return False

        for item in root_items:
            folder_name = item.get("name")
            
            # Skip hidden files or system placeholders
            if not folder_name or folder_name.startswith(".") or folder_name == ".emptyFolderPlaceholder":
                continue
            
            # Step 2: List contents INSIDE the student folder (e.g., 2400102415/)
            # This is necessary because 'deep=True' is often unreliable in the SDK
            print(f"🔍 Checking folder: {folder_name}")
            sub_items_raw = storage_api.list(folder_name, options={"limit": 100})
            sub_items = _normalize_list_response(sub_items_raw)
            
            for file_entry in sub_items:
                file_name = file_entry.get("name")
                
                # Check if it's an image file
                if file_name and Path(file_name).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    
                    # Create the student-specific local directory
                    student_id = folder_name
                    remote_path = f"{student_id}/{file_name}"
                    
                    local_student_dir = local_path / student_id
                    local_student_dir.mkdir(parents=True, exist_ok=True)
                    local_file_path = local_student_dir / file_name

                    # Step 3: Download the image
                    try:
                        raw_data = storage_api.download(remote_path)
                        file_data = _download_bytes_from_response(raw_data)
                        
                        if file_data:
                            with open(local_file_path, "wb") as f:
                                f.write(file_data)
                            download_count += 1
                            print(f"   ✅ Saved: {local_file_path.relative_to(local_path.parent)}")
                        else:
                            print(f"   ⚠ Empty data for {remote_path}")
                    except Exception as e:
                        print(f"   ❌ Error downloading {remote_path}: {e}")

    except Exception as e:
        print(f"❌ Critical error during traversal: {e}")
        return False

    print(f"\n✨ Summary: Downloaded {download_count} images into {local_images_dir}")
    return download_count > 0