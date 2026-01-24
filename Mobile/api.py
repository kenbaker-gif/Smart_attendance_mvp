import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import cv2
import numpy as np
import os
import sys
import pickle
import time  # ✅ Needed for timers
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware

# --- 1. PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

env_path = project_root / "secrets.env"
load_dotenv(env_path)

# --- GLOBAL VARIABLES ---
last_update_time = 0      # Tracks when we last checked Supabase
last_file_version = ""    # ✅ Tracks the specific version of the file we have in RAM

# --- 2. SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Database Error: {e}")

# --- 3. ENGINE IMPORT ---
try:
    from app.face_engine.insightface_engine import verify_face, update_face_bank 
except ImportError:
    print("CRITICAL: Face engine could not load.")
    def update_face_bank(data): pass
    pass

# --- 4. SMART DATA LOADING LOGIC ---
async def fetch_and_update_encodings():
    """
    Checks Supabase Storage metadata. 
    Only downloads the .pkl file if it is newer than what we have in RAM.
    """
    global last_update_time
    global last_file_version  # We need to read/write this global

    if not supabase: return

    print("🔄 Smart-Refresh: Checking if file has changed in Storage...")
    try:
        # 1. List files to peek at metadata (timestamp/id)
        files_list = supabase.storage.from_("raw_faces").list("encodings")
        
        target_file = None
        target_metadata = None
        
        for f in files_list:
            if f['name'].endswith('.pkl') or f['name'].endswith('.pickle'):
                target_file = f['name']
                target_metadata = f
                break
        
        if not target_file:
            print("⚠️ Refresh: No .pkl file found.")
            return

        # 2. CHECK: Compare cloud version vs local version
        # We use 'updated_at' or 'id' as the version signature
        current_version = target_metadata.get('updated_at', '') # timestamp string
        
        if current_version and current_version == last_file_version:
            print("✅ File is unchanged. Skipping download.")
            # We still reset the timer so we don't check again for another 5 mins
            last_update_time = time.time()
            return

        # 3. If versions don't match, Download!
        print(f"⬇️ New version found ({current_version}). Downloading {target_file}...")
        file_path = f"encodings/{target_file}"
        data_bytes = supabase.storage.from_("raw_faces").download(file_path)
        
        data = pickle.loads(data_bytes)
        
        if "names" in data and "encodings" in data:
            names = data["names"]
            encodings = data["encodings"]
            
            new_knowledge_base = {str(name): enc for name, enc in zip(names, encodings)}
            
            # Update the AI Engine
            update_face_bank(new_knowledge_base)
            
            # ✅ Update our version trackers
            last_file_version = current_version
            last_update_time = time.time()
            
            print(f"✅ Loaded {len(new_knowledge_base)} students. RAM Updated.")
        else:
            print(f"❌ Format Error in {target_file}")

    except Exception as e:
        print(f"❌ Refresh Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 Server Starting...")
    await fetch_and_update_encodings() # Always load on "Cold Start"
    yield
    # --- SHUTDOWN ---
    print("🛑 Server Shutting Down...")

# --- 5. API APP ---
app = FastAPI(title="Attendance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. HELPER FUNCTIONS ---
def get_student_name(student_id: str) -> str:
    """Fetches real name from Supabase DB using the ID."""
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
        print(f"📝 Logged: {student_id} ({status})")
    except Exception as e:
        print(f"❌ Log Error: {e}")

# --- 7. ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/refresh")
async def manual_refresh():
    # Force a refresh regardless of timer
    await fetch_and_update_encodings()
    return {"status": "success"}

@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
    global last_update_time
    
    # ✅ LAZY CHECK: Is the 5-minute timer up?
    # If yes, we call fetch_and_update, which will check the file version.
    if time.time() - last_update_time > 300:
        print("⏰ Timer expired (>5 mins). Checking storage...")
        await fetch_and_update_encodings()

    # Read Image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Run AI
    try:
        result = verify_face(img_bgr)
    except Exception as e:
        print(f"Engine Error: {e}")
        return {"status": "error", "message": "Processing Error"}

    if not result:
        return {"status": "failed", "message": "No face detected"}

    # Extract Data
    confidence = result.get("confidence", 0.0)
    status = result.get("status", "failed")
    message = result.get("message", "Unknown Identity")
    
    bbox_raw = result.get("bbox")
    kps_raw = result.get("kps")
    bbox_list = bbox_raw if bbox_raw is not None else []
    kps_list = kps_raw if kps_raw is not None else []

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