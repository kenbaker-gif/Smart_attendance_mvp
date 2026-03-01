import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel
import cv2
import numpy as np
import os
import sys
import pickle
import time
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
last_update_time = 0      
last_file_version = ""    

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
    global last_update_time
    global last_file_version 

    if not supabase: return

    print("🔄 Smart-Refresh: Checking if file has changed in Storage...")
    try:
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

        current_version = target_metadata.get('updated_at', '') 
        
        if current_version and current_version == last_file_version:
            print("✅ File is unchanged. Skipping download.")
            last_update_time = time.time()
            return

        print(f"⬇️ New version found ({current_version}). Downloading {target_file}...")
        file_path = f"encodings/{target_file}"
        data_bytes = supabase.storage.from_("raw_faces").download(file_path)
        
        data = pickle.loads(data_bytes)
        
        if "names" in data and "encodings" in data:
            names = data["names"]
            encodings = data["encodings"]
            new_knowledge_base = {str(name): enc for name, enc in zip(names, encodings)}
            update_face_bank(new_knowledge_base)
            last_file_version = current_version
            last_update_time = time.time()
            print(f"✅ Loaded {len(new_knowledge_base)} students. RAM Updated.")
        else:
            print(f"❌ Format Error in {target_file}")

    except Exception as e:
        print(f"❌ Refresh Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting...")
    await fetch_and_update_encodings()
    yield
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
    if not supabase: return student_id
    try:
        resp = supabase.table("students").select("name").eq("id", student_id).execute()
        if resp.data: return resp.data[0]['name']
    except: pass
    return student_id

def log_attendance(student_id: str, confidence: float, status: str):
    if not supabase: return

    # ✅ Extract institution_id from prefixed student_id
    # e.g. NKU2400102435 → NKU, MUK2400102435 → MUK
    # Look up institution_id from students table using student_id
    institution_id = None
    if student_id and status == "success" and supabase:
        try:
            resp = supabase.table("students").select("institution_id").eq("id", student_id).maybeSingle().execute()
            if resp.data:
                institution_id = resp.data.get("institution_id")
        except:
            pass

    data = {
        "student_id":       student_id if status == "success" else None,
        "confidence":       float(confidence),
        "detection_method": "mobile_api",
        "verified":         status,
        "institution_id":   institution_id,
    }
    try:
        supabase.table('attendance_records').insert(data).execute()
        print(f"📝 Logged: {student_id} | {institution_id} | {status}")
    except Exception as e:
        print(f"❌ Background Log Error: {e}")

# --- 7. ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/refresh")
async def manual_refresh():
    await fetch_and_update_encodings()
    return {"status": "success"}

@app.post("/verify")
async def verify_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    global last_update_time
    
    if time.time() - last_update_time > 300:
        print("⏰ Timer expired (>5 mins). Checking storage...")
        await fetch_and_update_encodings()

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        result = verify_face(img_bgr)
    except Exception as e:
        print(f"Engine Error: {e}")
        return {"status": "error", "message": "Processing Error"}

    if not result:
        return {"status": "failed", "message": "No face detected"}

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
        background_tasks.add_task(log_attendance, student_id, confidence, "success")
        return {
            "status": "success",
            "student_id": student_id,
            "name": real_name,
            "confidence": round(confidence, 2),
            "bbox": bbox_list,
            "kps": kps_list
        }
    else:
        background_tasks.add_task(log_attendance, "Unknown", confidence, "failed")
        return {
            "status": "failed",
            "message": message,
            "confidence": round(confidence, 2),
            "bbox": bbox_list,
            "kps": kps_list
        }