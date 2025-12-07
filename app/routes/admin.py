from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from supabase import create_client
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables (Important: Ensure this runs if this file is imported first)
load_dotenv()

# --- 1. ADMIN SETUP (Security and Supabase Client) ---

# Get credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

# Initialize Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not found in environment variables.")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define security scheme
security = HTTPBearer()

def verify_admin(token: str = Depends(security)):
    """Dependency to check if the provided token matches the ADMIN_SECRET."""
    if token.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Key")
    # Return True or any value to indicate success (FastAPI usually handles the return value)
    return True 

# --- 2. ROUTER DEFINITION ---

# Define the router with the /admin prefix
# This ensures all routes defined below start with /admin
router = APIRouter(
    prefix="/admin", 
    tags=["Admin & Reports"]
)

# --- 3. ENDPOINTS (Moved from main.py) ---

@router.get("/attendance", response_model=List[Dict[str, Any]])
def get_attendance(admin: bool = Depends(verify_admin)):
    """Fetches all raw attendance records."""
    # Note: Using the corrected table name 'attendance_records'
    response = supabase.table("attendance_records").select("*").execute()
    return response.data

@router.get("/attendance_summary", response_model=Dict[str, Any])
def get_summary(admin: bool = Depends(verify_admin)):
    """Calculates and returns summary statistics for attendance."""
    # Note: Using the corrected table name 'attendance_records'
    response = supabase.table("attendance_records").select("*").execute()
    rows = response.data

    total_present = sum(1 for r in rows if r["status"] == "present")
    total_absent = sum(1 for r in rows if r["status"] == "absent")
    
    by_student = {}
    for r in rows:
        # Note: Using the corrected column name 'students'
        student_name = r["students"] 
        by_student[student_name] = by_student.get(student_name, 0) + (1 if r["status"] == "present" else 0)

    return {
        "total_present": total_present, 
        "total_absent": total_absent, 
        "by_student": by_student
    }