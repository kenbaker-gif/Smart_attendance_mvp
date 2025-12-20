import shutil
from pathlib import Path
from typing import List, Union
from supabase import create_client

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

def download_all_supabase_images(
    supabase_url: str,
    supabase_key: str,
    supabase_bucket: str,
    local_images_dir: str,
    clear_local: bool = True,
) -> bool:
    """
    Downloads images from Supabase using the structure: StudentID/Image.jpg
    """
    local_path = Path(local_images_dir)
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        storage_api = supabase.storage.from_(supabase_bucket)
    except Exception as e:
        print(f"❌ Supabase Client Error: {e}")
        return False

    # 1. Prepare local directory
    if clear_local and local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    download_count = 0
    downloaded_ids = []

    try:
        # 2. List the root to find Student ID folders
        root_items = _normalize_list_response(storage_api.list("", options={"limit": 1000}))
        
        # Filter out hidden files like .emptyFolderPlaceholder
        folder_names = [item['name'] for item in root_items if not item['name'].startswith('.')]

        for student_id in folder_names:
            # 3. List inside each Student ID folder
            sub_items = _normalize_list_response(storage_api.list(student_id))
            
            for file_entry in sub_items:
                file_name = file_entry.get("name")
                
                # Only grab image files
                if file_name and Path(file_name).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    remote_path = f"{student_id}/{file_name}"
                    
                    # Create local structure: raw_faces/2400102415/1.jpg
                    local_student_dir = local_path / student_id
                    local_student_dir.mkdir(parents=True, exist_ok=True)
                    local_file_path = local_student_dir / file_name

                    # 4. Perform Download
                    res = storage_api.download(remote_path)
                    data = _download_bytes_from_response(res)
                    
                    if data:
                        with open(local_file_path, "wb") as f:
                            f.write(data)
                        download_count += 1
                        if student_id not in downloaded_ids:
                            downloaded_ids.append(student_id)

        print(f"✅ Successfully downloaded {download_count} images for {len(downloaded_ids)} students.")
        return download_count > 0

    except Exception as e:
        print(f"❌ Critical error during download: {e}")
        return False