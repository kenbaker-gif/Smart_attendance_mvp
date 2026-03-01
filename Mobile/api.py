import os
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from supabase import create_client
import pickle
import time
import sys

load_dotenv()

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# --- CONFIG ---
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
LOG_DIR         = PROJECT_ROOT / "logs"
LOG_FILE        = LOG_DIR / "attendance.log"
for d in [DATA_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
ADMIN_SECRET      = os.getenv("ADMIN_SECRET", "")
USE_SUPABASE      = os.getenv("USE_SUPABASE", "false").lower() == "true"

# --- LOGGING ---
from app.utils.logger import logger
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

# --- SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Error: {e}")

# --- GLOBAL STATE ---
last_update_time  = 0
last_file_version = ""
_name_cache:        dict = {}
_institution_cache: dict = {}

# --- ENGINE ---
try:
    from app.face_engine.insightface_engine import verify_face, update_face_bank
except ImportError:
    print("CRITICAL: Face engine could not load.")
    def update_face_bank(data): pass
    def verify_face(img): return None

# --- RECOGNITION SERVICE (for sync) ---
from app.services.recognition import RecognitionService
recognition_service = None

# --- CACHE ---
async def preload_student_cache():
    global _name_cache, _institution_cache
    if not supabase: return
    try:
        resp = supabase.table("students").select("id, name, institution_id").execute()
        for s in resp.data:
            _name_cache[s['id']]        = s.get('name', s['id'])
            _institution_cache[s['id']] = s.get('institution_id')
        print(f"✅ Preloaded {len(_name_cache)} students into cache")
    except Exception as e:
        print(f"❌ Cache preload failed: {e}")

# --- ENCODINGS ---
async def fetch_and_update_encodings():
    global last_update_time, last_file_version
    if not supabase: return
    print("🔄 Smart-Refresh: Checking if file has changed...")
    try:
        files_list = supabase.storage.from_("raw_faces").list("encodings")
        target_file = target_metadata = None
        for f in files_list:
            if f['name'].endswith(('.pkl', '.pickle')):
                target_file = f['name']
                target_metadata = f
                break
        if not target_file:
            print("⚠️ No .pkl file found.")
            return
        current_version = target_metadata.get('updated_at', '')
        if current_version and current_version == last_file_version:
            print("✅ File unchanged. Skipping download.")
            last_update_time = time.time()
            return
        print(f"⬇️ Downloading {target_file}...")
        data_bytes = supabase.storage.from_("raw_faces").download(f"encodings/{target_file}")
        data = pickle.loads(data_bytes)
        if "names" in data and "encodings" in data:
            kb = {str(n): e for n, e in zip(data["names"], data["encodings"])}
            update_face_bank(kb)
            last_file_version = current_version
            last_update_time  = time.time()
            print(f"✅ Loaded {len(kb)} students.")
        else:
            print("❌ Format Error in pkl")
    except Exception as e:
        print(f"❌ Refresh Error: {e}")

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognition_service
    logger.info("🚀 Starting Smart Attendance API")
    recognition_service = RecognitionService(
        data_dir=DATA_DIR,
        model_name="buffalo_s",
        threshold=DEFAULT_THRESHOLD
    )
    await fetch_and_update_encodings()
    await preload_student_cache()
    yield
    logger.info("🛑 Shutting down")

# --- APP ---
app = FastAPI(title="Smart Attendance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPERS ---
def get_student_name(student_id: str) -> str:
    if student_id in _name_cache:
        return _name_cache[student_id]
    if not supabase: return student_id
    try:
        resp = supabase.table("students").select("name, institution_id") \
            .eq("id", student_id).maybeSingle().execute()
        if resp.data:
            _name_cache[student_id]        = resp.data.get('name', student_id)
            _institution_cache[student_id] = resp.data.get('institution_id')
            return _name_cache[student_id]
    except: pass
    return student_id

def get_institution_id(student_id: str):
    return _institution_cache.get(student_id)

def log_attendance(student_id: str, confidence: float, status: str):
    if not supabase: return
    institution_id = get_institution_id(student_id) if status == "success" else None
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
        print(f"❌ Log Error: {e}")

def verify_admin_token(authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Invalid admin secret")

# --- ENDPOINTS ---
@app.get("/")
def root():
    return {"status": "online", "service": "Smart Attendance API"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service_ready": recognition_service is not None and recognition_service.is_initialized()
    }

@app.post("/refresh")
async def manual_refresh():
    await fetch_and_update_encodings()
    await preload_student_cache()
    return {"status": "success"}

@app.post("/verify")
async def verify_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    global last_update_time
    if time.time() - last_update_time > 300:
        await fetch_and_update_encodings()

    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
    status     = result.get("status", "failed")
    message    = result.get("message", "Unknown Identity")
    bbox_list  = result.get("bbox") or []
    kps_list   = result.get("kps")  or []

    if status == "success":
        student_id = result.get("student_id", "Unknown")
        real_name  = get_student_name(student_id)
        background_tasks.add_task(log_attendance, student_id, confidence, "success")
        return {
            "status":     "success",
            "student_id": student_id,
            "name":       real_name,
            "confidence": round(confidence, 2),
            "bbox":       bbox_list,
            "kps":        kps_list,
        }
    else:
        background_tasks.add_task(log_attendance, "Unknown", confidence, "failed")
        return {
            "status":     "failed",
            "message":    message,
            "confidence": round(confidence, 2),
            "bbox":       bbox_list,
            "kps":        kps_list,
        }

# --- ADMIN ENDPOINTS (for Streamlit dashboard) ---
@app.get("/admin/attendance-records")
def get_attendance_records(
    institution_id: str = None,
    limit: int = 500,
    _=Depends(verify_admin_token)
):
    """Fetch attendance records — optionally filtered by institution."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase.table("attendance_records") \
            .select("*, students(name)") \
            .order("timestamp", desc=True) \
            .limit(limit)
        if institution_id:
            query = query.eq("institution_id", institution_id)
        return query.execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/attendance_summary")
def get_summary(institution_id: str = None, _=Depends(verify_admin_token)):
    """Summary stats for dashboard."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase.table("attendance_records").select("*")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        rows = query.execute().data

        total_present = sum(1 for r in rows if r.get("verified") == "success")
        total_absent  = sum(1 for r in rows if r.get("verified") == "failed")
        by_student    = {}
        for r in rows:
            sid = r.get("student_id") or "Unknown"
            if r.get("verified") == "success":
                by_student[sid] = by_student.get(sid, 0) + 1

        return {
            "total_present": total_present,
            "total_absent":  total_absent,
            "by_student":    by_student,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/students")
def get_students(institution_id: str = None, _=Depends(verify_admin_token)):
    """List students — optionally filtered by institution."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase.table("students").select("*").order("name")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        data = query.execute().data
        return {"students": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-encodings")
async def sync_encodings(authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    if not recognition_service or not USE_SUPABASE:
        raise HTTPException(status_code=400, detail="Service not configured")
    try:
        stats = recognition_service.sync_encodings()
        await preload_student_cache()
        return {"success": True, "message": "Sync complete", **stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)