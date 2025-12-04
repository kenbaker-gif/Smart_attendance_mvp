import shutil
from pathlib import Path
from typing import List, Union
from supabase import create_client

def _normalize_list_response(resp) -> List[dict]:
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

def download_all_supabase_images(
    supabase_url: str,
    supabase_key: str,
    supabase_bucket: str,
    local_images_dir: str,
    clear_local: bool = True,
) -> bool:

    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        return False

    local_path = Path(local_images_dir)
    storage_api = supabase.storage.from_(supabase_bucket)

    try:
        if clear_local and local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to prepare local directory: {e}")
        return False

    try:
        all_files_raw = storage_api.list("", options={"limit": 1000, "deep": True})
        all_files = _normalize_list_response(all_files_raw)
    except Exception as e:
        print(f"❌ Failed to list Supabase bucket: {e}")
        return False

    download_count = 0

    for file_entry in all_files:
        remote_path = file_entry.get("id") or file_entry.get("name")
        if not remote_path or remote_path.endswith("/"):
            continue

        filename = Path(remote_path).name
        if Path(filename).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        try:
            student_id = filename[:10]
            if not (len(student_id) == 10 and student_id.isdigit()):
                continue
        except IndexError:
            continue

        local_file_path = local_path / student_id / filename
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            raw_data = storage_api.download(remote_path)
            file_data = _download_bytes_from_response(raw_data)
            if file_data:
                with open(local_file_path, "wb") as f:
                    f.write(file_data)
                download_count += 1
        except Exception as e:
            print(f"❌ Error downloading {remote_path}: {e}")

    print(f"✅ Downloaded {download_count} files to {local_images_dir}")
    return download_count > 0
