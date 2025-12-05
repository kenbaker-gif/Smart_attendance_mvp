"""
app/supabase_utils.py

Robust helpers for downloading images from Supabase storage into a nested
folder structure:

    data/raw_faces/<student_id>/<filename>

This module:
- supports deep listing
- normalizes various SDK return shapes
- normalizes download() responses to bytes
- can be used by backend scripts (retrain) or recognition pipeline
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

# Lazy import so module can be imported without env vars in some contexts
_supabase_client = None

def init_supabase_client(url: str, key: str):
    """Initialize and cache a Supabase client. Call once at startup."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    try:
        from supabase import create_client
    except Exception as e:
        raise RuntimeError(f"supabase package not available: {e}")
    _supabase_client = create_client(url, key)
    return _supabase_client


def _normalize_list_response(resp) -> List[dict]:
    """Turn various SDK responses into a list of file dicts with 'name' keys."""
    if resp is None:
        return []
    # Some SDK versions return list directly
    if isinstance(resp, list):
        return resp
    # Some return dict-like containers with 'data' or 'files'
    if isinstance(resp, dict):
        for k in ("data", "files", "list"):
            if k in resp and isinstance(resp[k], list):
                return resp[k]
        # If it has 'error' or unexpected structure, return empty
        return []
    # Fallback
    return []


def _download_bytes_from_response(res) -> Optional[bytes]:
    """Normalize download() return value to raw bytes or None."""
    if res is None:
        return None
    # raw bytes or bytearray
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    # Some SDKs return a dict with 'data' or 'body'
    if isinstance(res, dict):
        # Error -> None
        if res.get("error"):
            return None
        for key in ("data", "body", "content"):
            val = res.get(key)
            if isinstance(val, (bytes, bytearray)):
                return bytes(val)
            if isinstance(val, str):
                return val.encode()
        return None
    # If it's a file-like object
    try:
        if hasattr(res, "read"):
            return res.read()
    except Exception:
        pass
    # Try casting to bytes
    try:
        return bytes(res)
    except Exception:
        return None


def download_all_supabase_images(
    supabase_url: str,
    supabase_key: str,
    bucket_name: str,
    local_root: str = "data/raw_faces",
    clear_local: bool = False,
    limit: int = 5000,
    deep: bool = True,
) -> bool:
    """
    Download all files from a Supabase bucket and arrange them into:
      local_root/<student_id>/<original_filename>

    student_id is derived from the folder name if present (preferred) or,
    if bucket stores flat filenames like '2400102415_abc.jpg', the student_id
    extraction may be adjusted by your pipeline. This function assumes the
    bucket stores nested paths like '2400102415/1.jpg' (recommended for Option A).

    Returns True on success (>=1 files downloaded), False otherwise.
    """
    # sanity
    if not supabase_url or not supabase_key or not bucket_name:
        print("❌ Supabase credentials or bucket missing.")
        return False

    client = init_supabase_client(supabase_url, supabase_key)
    storage = client.storage.from_(bucket_name)

    local_root_path = Path(local_root)

    # Optionally clear local folder to ensure fresh data
    if clear_local and local_root_path.exists():
        try:
            shutil.rmtree(local_root_path)
        except Exception as e:
            print(f"⚠ Failed to clear local folder {local_root}: {e}")

    local_root_path.mkdir(parents=True, exist_ok=True)

    # List files (deep listing ensures nested paths are returned)
    try:
        options = {"limit": limit}
        if deep:
            options["deep"] = True
        raw_list = storage.list("", options)
        files = _normalize_list_response(raw_list)
    except Exception as e:
        print(f"❌ Failed to list bucket contents: {e}")
        return False

    if not files:
        print("⚠ No files found in bucket (or list returned empty).")
        return False

    download_count = 0

    for entry in files:
        # robustly get the path key
        remote_path = entry.get("name") or entry.get("id")
        if not remote_path:
            continue
        # skip folder placeholders
        if remote_path.endswith("/"):
            continue
        # Expect nested paths like '2400102415/1.jpg'
        parts = remote_path.split("/")
        if len(parts) != 2:
            # Skip unexpected shapes — user can adapt if their bucket is flat
            # (e.g., '2400102415_1.jpg') by post-processing or by calling a different helper.
            print(f"⚠ Skipping unexpected path shape: {remote_path}")
            continue

        student_id, filename = parts
        # optional validation of student_id (numeric / length) can be done by caller

        local_dir = local_root_path / student_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file_path = local_dir / filename

        try:
            raw = storage.download(remote_path)
            data = _download_bytes_from_response(raw)
            if data:
                with open(local_file_path, "wb") as fh:
                    fh.write(data)
                download_count += 1
                # small feedback
                print(f"⬇ Downloaded: {remote_path} -> {local_file_path}")
            else:
                print(f"⚠ Empty data for {remote_path} (possible RLS or access issue)")
        except Exception as e:
            print(f"❌ Failed to download {remote_path}: {e}")

    print(f"✅ Download complete. Total files downloaded: {download_count}")
    return download_count > 0
