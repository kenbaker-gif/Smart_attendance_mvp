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
import pickle  # ✅ Make sure to import this at the top of your file

import pickle # ✅ Ensure this is imported at the top

async def fetch_and_update_encodings():
    """Downloads the MASTER encoding file from Storage and updates the Engine."""
    if not supabase: 
        print("⚠️ Auto-Refresh: Supabase not connected.")
        return

    print("🔄 Auto-Refresh: Checking Storage for master encoding file...")
    
    try:
        # 1. Find the file name (We don't hardcode it, just in case it changes)
        files_list = supabase.storage.from_("raw_faces").list("encodings")
        
        # Look for the first file ending in .pkl or .pickle
        target_file = None
        for f in files_list:
            if f['name'].endswith('.pkl') or f['name'].endswith('.pickle'):
                target_file = f['name']
                break
        
        if not target_file:
            print("⚠️ Auto-Refresh: No .pkl file found in 'raw_faces/encodings'.")
            return

        print(f"⬇️ Downloading master file: {target_file}...")

        # 2. Download the file into memory (RAM)
        file_path = f"encodings/{target_file}"
        data_bytes = supabase.storage.from_("raw_faces").download(file_path)
        
        # 3. Open the file (Unpickle)
        # Expected structure: { "names": ["id1", "id2"], "encodings": [[...], [...]] }
        data = pickle.loads(data_bytes)
        
        # 4. Convert Data Structure
        # The Engine wants: { "ID": [Vector] }
        # The File has: Lists
        
        if "names" in data and "encodings" in data:
            names = data["names"]
            encodings = data["encodings"]
            
            # Zip them together into a Dictionary
            new_knowledge_base = {
                str(name): enc 
                for name, enc in zip(names, encodings)
            }
            
            # 5. Send to Engine
            update_face_bank(new_knowledge_base)
            print(f"✅ Auto-Refresh: Successfully unpacked {len(new_knowledge_base)} students from {target_file}.")
            
        else:
            print(f"❌ Error: {target_file} has wrong format. Keys found: {data.keys()}")

    except Exception as e:
        print(f"❌ Auto-Refresh Critical Error: {e}")

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