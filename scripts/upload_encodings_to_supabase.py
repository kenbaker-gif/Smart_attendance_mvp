#!/usr/bin/env python3
import os
import sys
from pathlib import Path

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
    print("❌ Missing SUPABASE_URL, SUPABASE_KEY or SUPABASE_BUCKET environment variables.")
    sys.exit(2)

try:
    from supabase import create_client
except Exception as e:
    print(f"❌ Could not import supabase client: {e}")
    sys.exit(3)

ENCODINGS_PATH = Path("streamlit/data/encodings_insightface.pkl")
if not ENCODINGS_PATH.exists():
    print(f"❌ Encodings file not found at {ENCODINGS_PATH}")
    sys.exit(4)

client = create_client(SUPABASE_URL, SUPABASE_KEY)
remote_path = os.getenv("ENCODINGS_REMOTE_PATH", "encodings/encodings_insightface.pkl")

# Remove existing remote file (best-effort)
try:
    client.storage.from_(SUPABASE_BUCKET).remove([remote_path])
    print(f"Removed existing object {remote_path} (if present)")
except Exception:
    pass

# Upload
try:
    with open(ENCODINGS_PATH, "rb") as fh:
        data = fh.read()
    
    # Use "no-cache" so your Streamlit app sees the update immediately
    client.storage.from_(SUPABASE_BUCKET).upload(
        path=remote_path, 
        file=data, 
        file_options={"cacheControl": "0", "contentType": "application/octet-stream"}
    )
    
    print(f"✅ Uploaded {ENCODINGS_PATH} to {SUPABASE_BUCKET}/{remote_path}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Failed to upload encodings: {e}")
    sys.exit(5)