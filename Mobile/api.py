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

# --- 1. PATH SETUP (CRITICAL) ---
# Ensure we can find the 'app' folder from the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

env_path = project_root / "secrets.env"
load_dotenv(env_path)

# --- 2. INITIALIZE API ---
app = FastAPI(title="Attendance API")

# --- 3. ADD MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. SUPABASE SETUP ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Database Error: {e}")

# --- 5. IMPORT ENGINE ---
try:
    from app.face_engine.insightface_engine import verify_face
except ImportError:
    print("CRITICAL: Face engine could not load. Check sys.path.")
    pass

# --- HELPER FUNCTIONS ---
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

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "service": "Face Recognition API"}

@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
    """
    Receives an image file, processes it, and returns the student identity + coordinates.
    """
    # 1. Read Image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 2. Run AI
    try:
        result = verify_face(img_bgr)
    except Exception as e:
        print(f"Engine Error: {e}")
        return {"status": "error", "message": "Server Processing Error", "confidence": 0.0}

    # 3. Handle Result
    if not result:
        return {"status": "failed", "message": "No face detected", "confidence": 0.0}

    # Extract values safely
    confidence = result.get("confidence", 0.0)
    status = result.get("status", "failed")
    message = result.get("message", "Unknown Identity")

    # --- NEW: Extract Coordinates for Flutter Overlay ---
    # We must convert numpy arrays to lists so JSON doesn't crash
    bbox_raw = result.get("bbox")
    kps_raw = result.get("kps")
    
    bbox_list = bbox_raw.tolist() if bbox_raw is not None else []
    kps_list = kps_raw.tolist() if kps_raw is not None else []

    if status == "success":
        student_id = result.get("student_id", "Unknown")
        real_name = get_student_name(student_id)

        # Log it
        log_attendance(student_id, confidence, "success")

        return {
            "status": "success",
            "student_id": student_id,
            "name": real_name,
            "confidence": round(confidence, 2),
            # ✅ Send coordinates to phone
            "bbox": bbox_list,
            "kps": kps_list
        }
    else:
        # Log failure safely
        log_attendance("Unknown", confidence, "failed")
        return {
            "status": "failed",
            "message": message,
            "confidence": round(confidence, 2),
            # ✅ Send coordinates (so we can draw Red Box)
            "bbox": bbox_list,
            "kps": kps_list
        }