from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Setup Paths & Env FIRST ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

env_path = project_root / "secrets.env"
load_dotenv(env_path)

# --- 2. Initialize API App Object ---
app = FastAPI(title="Attendance API")

# --- 3. Add Middleware (NOW 'app' is defined!) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Supabase Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Database Error: {e}")

# --- 5. Import Your Engine ---
try:
    from app.face_engine.insightface_engine import verify_face
except ImportError:
    print("CRITICAL: Face engine could not load.")
    sys.exit(1)

# --- Helper Functions (Reused) ---
def get_student_name(student_id: str) -> str:
    if not supabase: return student_id
    try:
        resp = supabase.table("students").select("name").eq("id", student_id).execute()
        if resp.data: return resp.data[0]['name']
    except: pass
    return student_id

def log_attendance(student_id: str, confidence: float, status: str):
    if not supabase: return
    data = {
        "student_id": student_id if status == "success" else None,
        "confidence": float(confidence),
        "detection_method": "mobile_api",
        "verified": status
    }
    try:
        supabase.table('attendance_records').insert(data).execute()
    except Exception as e:
        print(f"Log Error: {e}")

# --- API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "online", "service": "Face Recognition API"}

@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = verify_face(img_bgr)

    if not result:
        return {"status": "failed", "message": "No face detected", "confidence": 0.0}

    if result["status"] == "success":
        student_id = result["student_id"]
        real_name = get_student_name(student_id)
        log_attendance(student_id, result["confidence"], "success")
        return {
            "status": "success",
            "student_id": student_id,
            "name": real_name,
            "confidence": round(result["confidence"], 2)
        }
    else:
        log_attendance("Unknown", result["confidence"], "failed")
        return {
            "status": "failed",
            "message": "Unknown Person", 
            "confidence": round(result["confidence"], 2)
        }