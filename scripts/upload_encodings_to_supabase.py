import os
import sys
from pathlib import Path
from supabase import create_client

def upload_to_supabase():
    # 1. Load credentials from environment
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    bucket_name = os.getenv("SUPABASE_BUCKET", "raw_faces")
    remote_path = "encodings/encodings_insightface.pkl"

    if not url or not key:
        print("❌ Error: SUPABASE_URL or SUPABASE_KEY not set.")
        sys.exit(1)

    # 2. Smart Path Discovery
    # This looks for the pkl file anywhere in your project root
    possible_files = list(Path(".").rglob("encodings_insightface.pkl"))
    
    if not possible_files:
        print("❌ Error: Could not find 'encodings_insightface.pkl' anywhere in the workspace.")
        sys.exit(1)
    
    local_file_path = possible_files[0]
    print(f"📦 Found local file: {local_file_path}")

    # 3. Initialize and Upload
    try:
        supabase = create_client(url, key)
        
        with open(local_file_path, 'rb') as f:
            # We use upsert=True so it overwrites the old version
            res = supabase.storage.from_(bucket_name).upload(
                path=remote_path,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        
        print(f"✅ SUCCESS: Uploaded to {bucket_name}/{remote_path}")
        
    except Exception as e:
        # Check if error is because file already exists (some SDK versions)
        if "already exists" in str(e).lower():
            print("🔄 File exists. Attempting update...")
            with open(local_file_path, 'rb') as f:
                supabase.storage.from_(bucket_name).update(path=remote_path, file=f)
            print("✅ SUCCESS: Updated existing file.")
        else:
            print(f"💥 Upload failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    upload_to_supabase()