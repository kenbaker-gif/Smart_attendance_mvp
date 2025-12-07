# app/routes/attendance.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException # Added Depends, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer # NEW: Needed for verify_admin
import os
from dotenv import load_dotenv # NEW: Needed for os.getenv
from supabase import create_client

# Load environment variables here if necessary, though main.py loads them too.
load_dotenv() 

# Import database methods and models (already done, good)
from app.database import (
    # ... your existing imports ...
)
from app.models import Student, AttendanceRecord

# --- ADMIN SETUP (MOVED FROM main.py) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

security = HTTPBearer()
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

def verify_admin(token: str = Depends(security)):
    if token.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
# ----------------------------------------


# Define the router with the /admin prefix, so ALL routes here start with /admin
router = APIRouter(
    prefix="/admin", # NEW: Added the prefix here
    tags=["Attendance & Admin"]
)


@router.post("/capture/{student_id}") # THIS PATH BECOMES /admin/capture/{student_id}
async def capture_and_upload(student_id: str, file: UploadFile = File(...)):
    # ... your existing capture_and_upload logic ...
    # This route is now unnecessarily secured under the /admin prefix. 
    # It would be better to put this in a separate router, but we'll keep it here for now.
    # ...

    return RedirectResponse(url="/", status_code=303)


# --- ADMIN ENDPOINTS (MOVED FROM main.py) ---

@router.get("/attendance") # Final accessible path: /admin/attendance
def get_attendance(admin: bool = Depends(verify_admin)):
    # Assuming 'attendance_records' and 'students' are correct column names
    response = supabase.table("attendance_records").select("*").execute()
    return response.data

@router.get("/attendance_summary") # Final accessible path: /admin/attendance_summary
def get_summary(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    rows = response.data

    total_present = sum(1 for r in rows if r["status"] == "present")
    total_absent = sum(1 for r in rows if r["status"] == "absent")
    
    by_student = {}
    for r in rows:
        by_student[r["students"]] = by_student.get(r["students"], 0) + (1 if r["status"] == "present" else 0)

    return {"total_present": total_present, "total_absent": total_absent, "by_student": by_student}