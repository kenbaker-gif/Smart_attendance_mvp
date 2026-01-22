import asyncio
from contextlib import asynccontextmanager
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
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

env_path = project_root / "secrets.env"
load_dotenv(env_path)

# --- 4. SUPABASE SETUP (Moved up so it's available for lifespan) ---
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
    # We import the verification function AND the update function
    from app.face_engine.insightface_engine import verify_face, update_face_bank 
except ImportError:
    print("CRITICAL: Face engine could not load. Check sys.path.")
    def update_face_bank(data): pass # Placeholder to prevent crash if missing
    pass

# --- 6. AUTO-REFRESH LOGIC (NEW) ---
async def fetch_and_update_encodings():
    """Fetches embeddings from Supabase and pushes them to the Engine."""
    if not supabase: 
        print("⚠️ Auto-Refresh: Supabase not connected.")
        return

    print("🔄 Auto-Refresh: Fetching student list from Supabase...")
    try:
        # ✅ FIX 1: Selecting 'id' instead of 'student_id'
        resp = supabase.table("students").select("id, embedding").execute()
        data = resp.data
        
        if data:
            new_knowledge_base = {}
            for student in data:
                if student.get('embedding'):
                    # ✅ FIX 2: Using 'id' here as well
                    s_id = student.get('id') 
                    
                    # Ensure embedding is a list (Supabase returns JSON/List)
                    new_knowledge_base[s_id] = student['embedding']
            
            # Update the Engine's memory
            update_face_bank(new_knowledge_base)
            print(f"✅ Auto-Refresh: Successfully loaded {len(new_knowledge_base)} students.")
        else:
            print("⚠️ Auto-Refresh: No students found in database.")
            
    except Exception as e:
        print(f"❌ Auto-Refresh Error: {e}")

async def run_periodic_refresh():
    """Background task that runs every 5 minutes"""
    while True:
        await asyncio.sleep(300) # Wait 300 seconds (5 minutes)
        await fetch_and_update_encodings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 Server Starting: Loading Data...")
    await fetch_and_update_encodings() # Load immediately
    asyncio.create_task(run_periodic_refresh()) # Start background timer
    
    yield # App runs here
    
    # --- SHUTDOWN ---
    print("🛑 Server Shutting Down...")

# --- 2. INITIALIZE API (With Lifespan) ---
app = FastAPI(title="Attendance API", lifespan=lifespan)

# --- 3. ADD MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER FUNCTIONS ---
def get_student_name(student_id: str) -> str:
    if not supabase: return student_id
    try:
        resp = supabase.table("students").select("name").eq("student_id", student_id).execute()
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

# New Endpoint to force refresh immediately manually
@app.post("/refresh")
async def manual_refresh():
    await fetch_and_update_encodings()
    return {"status": "success", "message": "Database re-synced"}

@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
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

    confidence = result.get("confidence", 0.0)
    status = result.get("status", "failed")
    message = result.get("message", "Unknown Identity")

    bbox_raw = result.get("bbox")
    kps_raw = result.get("kps")
    
    bbox_list = bbox_raw.tolist() if bbox_raw is not None else []
    kps_list = kps_raw.tolist() if kps_raw is not None else []

    if status == "success":
        student_id = result.get("student_id", "Unknown")
        real_name = get_student_name(student_id)
        log_attendance(student_id, confidence, "success")

        return {
            "status": "success",
            "student_id": student_id,
            "name": real_name,
            "confidence": round(confidence, 2),
            "bbox": bbox_list,
            "kps": kps_list
        }
    else:
        log_attendance("Unknown", confidence, "failed")
        return {
            "status": "failed",
            "message": message,
            "confidence": round(confidence, 2),
            "bbox": bbox_list,
            "kps": kps_list
        }