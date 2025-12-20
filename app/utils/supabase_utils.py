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

    local_path = Path(local_images_dir)

    # If local images already exist and the caller does not want to clear them,
    # skip the Supabase download to avoid unnecessary network calls and avoid
    # overwriting local images.
    def _local_has_images(p: Path) -> bool:
        try:
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return True
        except Exception:
            return False
        return False

    if local_path.exists() and not clear_local and _local_has_images(local_path):
        print(f"⚠ Skipping Supabase download: local images already exist in {local_path}")
        return True

    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        return False

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

    # Debug: report how many objects we found
    try:
        print(f"🔍 Supabase list returned {len(all_files)} objects (showing up to 5): {all_files[:5]}")
    except Exception:
        pass

    download_count = 0

    for file_entry in all_files:
        remote_path = file_entry.get("id") or file_entry.get("name")
        if not remote_path or str(remote_path).endswith("/"):
            continue

        filename = Path(str(remote_path)).name
        if Path(filename).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # Extract a 10-digit student id from the full remote path if possible
        student_id = None
        try:
            import re
            m = re.search(r"(\d{10})", str(remote_path))
            if m:
                student_id = m.group(1)
        except Exception:
            student_id = None

        if not student_id:
            # Fallback: try filename prefix
            try:
                candidate = filename[:10]
                if len(candidate) == 10 and candidate.isdigit():
                    student_id = candidate
            except Exception:
                pass

        if not student_id:
            # Could not infer student id from path/filename -> skip but log for debugging
            print(f"⚠ Skipping {remote_path}: could not infer 10-digit student id from path or filename")
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
            else:
                print(f"⚠ No data returned for {remote_path}")
        except Exception as e:
            print(f"❌ Error downloading {remote_path}: {e}")

    print(f"✅ Downloaded {download_count} files to {local_images_dir}")
    if download_count == 0 and len(all_files) == 0:
        print("⚠ No objects were found in the Supabase bucket. Confirm the bucket name and that it contains image files.")
    elif download_count == 0:
        print("⚠ No image files matching the expected patterns were downloaded. Check naming conventions (student ID in filename or path).")

    return download_count > 0
