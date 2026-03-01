import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
_name_cache: dict = {}       # student_id → name
_institution_cache: dict = {}  # student_id → institution_id

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

# --- 4. PRELOAD STUDENT CACHE ---
async def preload_student_cache():
    """Load all student names + institution_ids into RAM on startup."""
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

# --- 5. SMART ENCODINGS REFRESH ---
async def fetch_and_update_encodings():
    global last_update_time, last_file_version
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

# --- 6. LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting...")
    await fetch_and_update_encodings()
    await preload_student_cache()   # ✅ preload names into RAM
    yield
    print("🛑 Server Shutting Down...")

# --- 7. APP ---
app = FastAPI(title="Attendance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 8. HELPER FUNCTIONS ---
def get_student_name(student_id: str) -> str:
    """Instant lookup from RAM cache — no DB call."""
    if student_id in _name_cache:
        return _name_cache[student_id]
    # Fallback to DB if not in cache (new student registered after startup)
    if not supabase: return student_id
    try:
        resp = supabase.table("students").select("name, institution_id") \
            .eq("id", student_id).maybeSingle().execute()
        if resp.data:
            _name_cache[student_id]        = resp.data.get('name', student_id)
            _institution_cache[student_id] = resp.data.get('institution_id')
            return _name_cache[student_id]
    except:
        pass
    return student_id

def get_institution_id(student_id: str) -> str | None:
    """Instant lookup from RAM cache — no DB call."""
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
        print(f"❌ Background Log Error: {e}")

# --- ADMIN AUTH ---
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

def verify_admin_token(authorization: str = None):
    from fastapi import Header
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Invalid admin secret")

# --- 9. ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online"}

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
    status     = result.get("status", "failed")
    message    = result.get("message", "Unknown Identity")
    bbox_list  = result.get("bbox") or []
    kps_list   = result.get("kps") or []

    if status == "success":
        student_id = result.get("student_id", "Unknown")
        real_name  = get_student_name(student_id)   # ✅ instant from cache
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
from fastapi import Depends, Header

def check_admin(authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Invalid admin secret")

@app.get("/admin/attendance-records")
def get_attendance_records(
    institution_id: str = None,
    limit: int = 500,
    _=Depends(check_admin)
):
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase.table("attendance_records") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(limit)
        if institution_id:
            query = query.eq("institution_id", institution_id)
        return query.execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/attendance_summary")
def get_summary(institution_id: str = None, _=Depends(check_admin)):
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase.table("attendance_records").select("*")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        rows = query.execute().data
        total_present = sum(1 for r in rows if r.get("verified") == "success")
        total_absent  = sum(1 for r in rows if r.get("verified") == "failed")
        by_student = {}
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
def get_students(institution_id: str = None, _=Depends(check_admin)):
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
    try:
        await fetch_and_update_encodings()
        await preload_student_cache()
        return {"success": True, "message": "Sync complete"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))