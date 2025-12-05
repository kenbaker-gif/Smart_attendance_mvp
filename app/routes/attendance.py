# app/routes/attendance.py

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import RedirectResponse
import os
from supabase import create_client

from app.database import (
    SessionLocal,
    add_attendance_record,
    get_student_attendance,
    get_today_attendance,
    get_attendance_summary,
    register_student
)
from app.models import Student, AttendanceRecord

router = APIRouter()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@router.post("/capture/{student_id}")
async def capture_and_upload(student_id: str, file: UploadFile = File(...)):
    """
    Upload student image directly to Supabase bucket.
    After completion, redirect to the homepage.
    """
    file_name = f"{student_id}_{file.filename}"

    try:
        file_bytes = await file.read()
        # Delete if exists (optional)
        try:
            supabase.storage.from_(SUPABASE_BUCKET).remove([f"{student_id}/{file_name}"])
        except Exception:
            pass  # Ignore if file does not exist

        # Upload file
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            f"{student_id}/{file_name}", file_bytes, {"cacheControl": "3600"}
        )

    except Exception as e:
        return {"error": f"Failed to upload image to Supabase: {e}"}

    return RedirectResponse(url="/", status_code=303)
