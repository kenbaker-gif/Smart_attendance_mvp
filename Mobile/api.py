import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Header, Form
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
from typing import Optional
from collections import defaultdict
from fastapi.concurrency import run_in_threadpool

# --- 1. PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

env_path = project_root / "secrets.env"
load_dotenv(env_path)

# --- GLOBAL VARIABLES ---
last_update_time = 0
last_file_version = ""
_name_cache: dict = {}
_institution_cache: dict = {}

# --- 2. SUPABASE ---
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_KEY         = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase       = None
supabase_admin = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Database Error: {e}")

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        print(f"Admin DB Error: {e}")

# --- 3. ENGINE IMPORT ---
try:
    from app.face_engine.insightface_engine import verify_face, update_face_bank, preload_models
except ImportError:
    print("CRITICAL: Face engine could not load.")
    def update_face_bank(data): pass
    def preload_models(): pass

# --- 4. HELPERS ---

def _bool_flag(value):
    """Safely coerce any truthy DB value to a Python bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


# --- 5. AUTH DEPENDENCIES ---

async def verify_supabase_token(authorization: str = Header(None)):
    """Verify that the request comes from a valid authenticated Supabase user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_response.user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")

async def check_admin(authorization: str = Header(None)):
    """
    Verify that the user is authenticated and is an admin.
    Accepts: is_admin=True OR is_super_admin=True OR role in ('admin', 'super_admin').
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user_id = user_response.user.id

        resp = supabase_admin.table("profiles") \
            .select("is_admin, is_super_admin, role") \
            .eq("id", user_id).limit(1).execute()

        profile = resp.data[0] if resp.data else None
        if not profile:
            raise HTTPException(status_code=403, detail="Profile not found")

        is_admin       = _bool_flag(profile.get("is_admin"))
        is_super_admin = _bool_flag(profile.get("is_super_admin"))
        role           = profile.get("role", "")

        if not (is_admin or is_super_admin or role in ("admin", "super_admin")):
            raise HTTPException(status_code=403, detail="Admin access required")

        return user_response.user
    except HTTPException:
        raise
    except Exception as e:
        print(f"[check_admin] error: {e!r}")
        raise HTTPException(status_code=401, detail="Token verification failed")

# --- 6. PRELOAD STUDENT CACHE ---
async def preload_student_cache():
    global _name_cache, _institution_cache
    if not supabase_admin:
        return
    try:
        _name_cache.clear()
        _institution_cache.clear()
        resp = supabase_admin.table("students").select("id, name, institution_id").execute()
        for s in resp.data:
            _name_cache[s['id']]        = s.get('name', s['id'])
            _institution_cache[s['id']] = s.get('institution_id')
        print(f"✅ Preloaded {len(_name_cache)} students into cache")
    except Exception as e:
        print(f"❌ Cache preload failed: {e}")

# --- 7. SMART ENCODINGS REFRESH ---
async def fetch_and_update_encodings():
    global last_update_time, last_file_version
    if not supabase_admin:
        return False

    print("🔄 Smart-Refresh: Checking if file has changed in Storage...")
    try:
        files_list = supabase_admin.storage.from_("raw_faces").list("encodings")

        target_file     = None
        target_metadata = None
        for f in files_list:
            if f['name'].endswith('.pkl') or f['name'].endswith('.pickle'):
                target_file     = f['name']
                target_metadata = f
                break

        if not target_file:
            print("⚠️ Refresh: No .pkl file found.")
            return False

        current_version = target_metadata.get('updated_at', '')

        if current_version and current_version == last_file_version:
            print("✅ File is unchanged. Skipping download.")
            last_update_time = time.time()
            return True

        print(f"⬇️ New version found ({current_version}). Downloading {target_file}...")
        file_path  = f"encodings/{target_file}"
        data_bytes = supabase_admin.storage.from_("raw_faces").download(file_path)
        data       = pickle.loads(data_bytes)

        if "names" in data and "encodings" in data:
            names              = data["names"]
            encodings          = data["encodings"]
            new_knowledge_base = {str(name): enc for name, enc in zip(names, encodings)}
            update_face_bank(new_knowledge_base)
            last_file_version = current_version
            last_update_time  = time.time()
            print(f"✅ Loaded {len(new_knowledge_base)} students. RAM Updated.")
            return True
        else:
            print(f"❌ Format Error in {target_file}")
            return False

    except Exception as e:
        print(f"❌ Refresh Error: {e}")
        return False


async def build_encodings_from_storage():
    global last_file_version, last_update_time
    if not supabase_admin:
        return
    print("🔨 Building encodings from storage...")

    try:
        inst_resp          = supabase_admin.table("institutions").select("id").execute()
        known_institutions = [r["id"] for r in inst_resp.data]
        print(f"📋 Found institutions: {known_institutions}")
    except Exception as e:
        print(f"⚠️ Could not fetch institutions, falling back: {e}")
        known_institutions = ["NKU", "MUK"]

    # Collect all embeddings per student (multiple photos)
    student_embeddings: dict = defaultdict(list)

    for institution in known_institutions:
        try:
            student_folders = supabase_admin.storage.from_("raw_faces").list(institution)
        except:
            print(f"⚠️ No folder found for {institution}")
            continue

        for folder in student_folders:
            student_id  = folder["name"]
            folder_path = f"{institution}/{student_id}"

            try:
                files = supabase_admin.storage.from_("raw_faces").list(folder_path)
            except:
                continue

            for f in files:
                if not f["name"].endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    img_bytes = supabase_admin.storage.from_("raw_faces").download(
                        f"{folder_path}/{f['name']}"
                    )
                    nparr   = np.frombuffer(img_bytes, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue

                    from app.face_engine.insightface_engine import get_embedding
                    embedding = get_embedding(img_bgr)
                    if embedding is not None:
                        student_embeddings[student_id].append(embedding)
                        print(f"   ✅ {institution}/{student_id}/{f['name']}")
                except Exception as e:
                    print(f"   ❌ {folder_path}/{f['name']}: {e}")

    if student_embeddings:
        # Average all embeddings per student for best accuracy
        kb = {}
        for student_id, embs in student_embeddings.items():
            avg_emb = np.mean(embs, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)
            kb[student_id] = avg_emb

        # Save to storage
        pkl_data  = {"names": list(kb.keys()), "encodings": list(kb.values())}
        pkl_bytes = pickle.dumps(pkl_data)

        try:
            supabase_admin.storage.from_("raw_faces").remove(["encodings/encodings_insightface.pkl"])
        except:
            pass
        supabase_admin.storage.from_("raw_faces").upload(
            "encodings/encodings_insightface.pkl",
            pkl_bytes,
        )
        print(f"✅ Saved {len(kb)} students to storage")

        # Update RAM directly — do NOT call fetch_and_update_encodings after this
        update_face_bank(kb)
        print("✅ RAM updated")

        # Mark version so the 5-min timer refresh skips re-downloading this same pkl
        try:
            files_list = supabase_admin.storage.from_("raw_faces").list("encodings")
            for f in files_list:
                if f['name'].endswith('.pkl'):
                    last_file_version = f.get('updated_at', '')
                    break
        except:
            pass
        last_update_time = time.time()

    else:
        print("⚠️ No embeddings generated — no face images found in storage")

# --- 8. LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server Starting...")
    preload_models()
    loaded = await fetch_and_update_encodings()  # smart: skips if pkl unchanged
    if not loaded:
        await build_encodings_from_storage()     # only if no pkl exists yet
    await preload_student_cache()
    yield
    print("🛑 Server Shutting Down.")

# --- 9. APP ---
app = FastAPI(title="Attendance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faceattend.app",
        "https://www.faceattend.app",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://api.faceattend.app",
        "https://mvp.faceattend.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# --- 10. HELPER FUNCTIONS ---
def get_student_name(student_id: str) -> str:
    if student_id in _name_cache:
        return _name_cache[student_id]
    if not supabase_admin:
        return student_id
    try:
        resp = supabase_admin.table("students").select("name, institution_id") \
            .eq("id", student_id).limit(1).execute()
        if resp.data and len(resp.data) > 0:
            _name_cache[student_id]        = resp.data[0].get('name', student_id)
            _institution_cache[student_id] = resp.data[0].get('institution_id')
            return _name_cache[student_id]
    except:
        pass
    return student_id

def get_institution_id(student_id: str) -> str | None:
    return _institution_cache.get(student_id)

def log_attendance(student_id: str, confidence: float, status: str, institution_id: Optional[str] = None, course_unit_id=None):
    if not supabase_admin:
        return
    if status == "success":
        institution_id = get_institution_id(student_id) or institution_id
        # Guard: confirm student still exists in DB before inserting
        try:
            check = supabase_admin.table("students").select("id").eq("id", student_id).limit(1).execute()
            if not check.data:
                print(f"⚠️ Skipped log: student {student_id} not in DB (stale embedding)")
                return
        except Exception as e:
            print(f"⚠️ Student existence check failed: {e}")
            return

    data = {
        "student_id":       student_id if status == "success" else None,
        "confidence":       float(confidence),
        "detection_method": "mobile_api",
        "verified":         status,
        "institution_id":   institution_id,
        "course_unit_id":   course_unit_id,
    }
    try:
        supabase_admin.table('attendance_records').insert(data).execute()
        print(f"📝 Logged: {student_id} | {institution_id} | {status}")
    except Exception as e:
        print(f"❌ Background Log Error: {e}")

# --- 11. ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/refresh")
async def manual_refresh(user=Depends(check_admin)):
    await fetch_and_update_encodings()
    await preload_student_cache()
    return {"status": "success"}

@app.post("/verify")
async def verify_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    institution_id: Optional[str] = Form(None),
    course_unit_id: Optional[str] = Form(None),
    user=Depends(verify_supabase_token),
):
    global last_update_time

    # 1. Non-blocking Cache Refresh

    if time.time() - last_update_time > 300:
        print("⏰ Timer expired (>5 mins). Checking storage...")
        background_tasks.add_task(fetch_and_update_encodings) # Run in background so student doesn't wait

    # 2. Fast Image Reading
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 🚀 OPTIMIZATION: Resize image to 640px width if it's too large
    # This reduces CPU load by up to 70% for high-res mobile photos
    h, w = img_bgr.shape[:2]
    if w > 640:
        scaling = 640 / w
        img_bgr = cv2.resize(img_bgr, (640, int(h * scaling)))

    # 🧠 OPTIMIZATION: Use run_in_threadpool
    # This offloads the heavy InsightFace math to a separate thread
    # Allowing your 8vCPUs to handle multiple students at once
    try:
        result = await run_in_threadpool(verify_face, img_bgr)
    except Exception as e:
        print(f"Engine Error: {e}")
        return {"status": "error", "message": "Processing Error"}

    # --- LOGGING & RESPONSE LOGIC ---
    if not result:
        return {"status": "failed", "message": "No face detected"}

    status = result.get("status", "failed")
    confidence = result.get("confidence", 0.0)
    bbox_list = result.get("bbox", [])
    kps_list = result.get("kps", [])
    message = result.get("message", "No face detected")

    # Offload DB logging to background so student gets response INSTANTLY
    student_id = result.get("student_id", "Unknown") if status == "success" else "Unknown"
    log_status = status if status in ["success", "spoof"] else "failed"
    
    background_tasks.add_task(
        log_attendance, 
        student_id, 
        confidence, 
        log_status, 
        institution_id, 
        course_unit_id
    )

    if status == "success":
        return {
            "status": "success",
            "student_id": student_id,
            "name": get_student_name(student_id),
            "confidence": round(confidence, 2),
            "liveness_score": result.get("liveness_score", 1.0),
            "bbox": bbox_list,
            "kps": kps_list,
        }
        
    elif status == "spoof":
        return {
            "status":         "spoof",
            "message":        "Spoof detected. Please use your real face.",
            "liveness_score": result.get("liveness_score", 0.0),
            "confidence":     0.0,
            "bbox":           bbox_list,
            "kps":            kps_list,
        }
    else:
        return {
            "status":     "failed",
            "message":    message,
            "confidence": round(confidence, 2),
            "bbox":       bbox_list,
            "kps":        kps_list,
        }


# --- 12. ADMIN ENDPOINTS ---

@app.get("/admin/attendance-records")
async def get_attendance_records(
    institution_id: str = None,
    limit: int = 500,
    user=Depends(check_admin),
):
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase_admin.table("attendance_records") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(limit)
        if institution_id:
            query = query.eq("institution_id", institution_id)
        return query.execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/attendance_summary")
async def get_summary(
    institution_id: str = None,
    user=Depends(check_admin),
):
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query         = supabase_admin.table("attendance_records").select("*")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        rows          = query.execute().data
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
async def get_students(
    institution_id: str = None,
    user=Depends(check_admin),
):
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        query = supabase_admin.table("students").select("*").order("name")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        data = query.execute().data
        return {"students": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/sync-encodings")
async def sync_encodings(user=Depends(check_admin)):
    """
    Rebuild face encodings from raw storage images and reload into RAM.
    Called automatically by the upload service after the 4th photo is uploaded.
    """
    try:
        await build_encodings_from_storage()
        await preload_student_cache()
        return {"success": True, "message": "Sync complete"}
    except Exception as e:
        print(f"[sync-encodings] error: {e!r}")
        raise HTTPException(status_code=500, detail=str(e))