import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import RedirectResponse
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

router = APIRouter()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_next_image_number(student_id: str) -> int:
    """
    Find the next available image number for the student.
    """
    try:
        files_resp = supabase.storage.from_(SUPABASE_BUCKET).list(student_id)
        if files_resp is None or "error" in files_resp:
            return 1
        existing_numbers = []
        for f in files_resp:
            name = f.get("name") or f.get("id") or ""
            # Extract the number from filename like '2.jpg'
            base = os.path.splitext(name)[0].split("_")[-1]
            if base.isdigit():
                existing_numbers.append(int(base))
        return max(existing_numbers, default=0) + 1
    except Exception:
        return 1

@router.post("/capture/{student_id}")
async def capture_and_upload(student_id: str, file: UploadFile = File(...)):
    """
    Upload student image directly to Supabase bucket.
    Filename is incremental (1.jpg, 2.jpg, …) inside the student's folder.
    """
    try:
        # 1. List existing files
        existing_files_resp = supabase.storage.from_(SUPABASE_BUCKET).list(student_id)
        existing_files = [f['name'] for f in existing_files_resp if f.get('name')]

        # 2. Determine next number
        nums = []
        for f in existing_files:
            try:
                num = int(Path(f).stem)  # filename without extension
                nums.append(num)
            except ValueError:
                continue
        next_num = max(nums, default=0) + 1

        file_name = f"{next_num}.jpg"

        # 3. Read file bytes
        file_bytes = await file.read()

        # 4. Upload to Supabase
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            f"{student_id}/{file_name}", file_bytes, {"cacheControl": "3600"}
        )

    except Exception as e:
        return {"error": f"Failed to upload image to Supabase: {e}"}

    return RedirectResponse(url="/", status_code=303)
