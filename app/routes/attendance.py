# app/routes/attendance.py

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import RedirectResponse
import os
from shutil import copyfileobj

from app.utils.face_utils import preprocess_faces
from app.database import (
    SessionLocal,
    add_attendance_record,
    get_student_attendance,
    get_today_attendance,
    get_attendance_summary,
    register_student
)
from app.models import Student, AttendanceRecord  # ✅ Corrected import path

router = APIRouter()

@router.post("/capture/{student_id}")
async def capture_and_preprocess(
    student_id: str,
    file: UploadFile = File(...),
    preprocess: bool = True
):
    """
    Save uploaded image for a student and optionally preprocess it immediately.
    After completion, redirect to the homepage.
    """
    # Step 1: Save raw image
    raw_dir = os.path.join("data", "raw_faces", student_id)
    os.makedirs(raw_dir, exist_ok=True)

    existing_files = os.listdir(raw_dir)
    new_index = len(existing_files) + 1
    file_ext = os.path.splitext(file.filename)[1]
    raw_path = os.path.join(raw_dir, f"{new_index}{file_ext}")

    try:
        with open(raw_path, "wb") as buffer:
            copyfileobj(file.file, buffer)
    except Exception as e:
        return {"error": f"Failed to save image: {e}"}

    # Step 2: Preprocess immediately if requested
    if preprocess:
        try:
            preprocess_faces()  # This will process all images in raw_faces
        except Exception as e:
            return {"error": f"Preprocessing failed: {e}"}

    # Step 3: Redirect back to homepage
    return RedirectResponse(url="/", status_code=303)