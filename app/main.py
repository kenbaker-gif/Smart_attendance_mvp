# main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
from app.routes.attendance import router as attendance_router
from supabase import create_client
import os
from dotenv import load_dotenv
import os

load_dotenv()


app = FastAPI(title="Smart Attendance System")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))

# Healthcheck endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Include attendance router
app.include_router(attendance_router, prefix="/attendance", tags=["Attendance"])

# --- Admin Panel Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

security = HTTPBearer()
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

def verify_admin(token: str = Depends(security)):
    if token.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.get("/admin/attendance")
def get_attendance(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    return response.data

@app.get("/admin/attendance_summary")
def get_summary(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    rows = response.data

    total_present = sum(1 for r in rows if r["status"] == "present")
    total_absent = sum(1 for r in rows if r["status"] == "absent")
    
    by_student = {}
    for r in rows:
        by_student[r["students"]] = by_student.get(r["students"], 0) + (1 if r["status"] == "present" else 0)

    return {"total_present": total_present, "total_absent": total_absent, "by_student": by_student}
