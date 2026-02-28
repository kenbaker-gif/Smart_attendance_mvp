import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Attendance — Upload Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET        = "raw_faces"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/")
def home():
    return {"status": "Smart Attendance: OTA ACTIVE"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-student-face")
async def upload_student(
    student_id:     str        = Form(...),
    name:           str        = Form(...),
    institution_id: str        = Form(...),   # ✅ NKU or MUK
    file:           UploadFile = File(...),
):
    # ── Validate inputs ────────────────────────────────────────────────
    if not student_id.strip():
        raise HTTPException(status_code=400, detail="student_id cannot be empty.")
    if not name.strip():
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    if not institution_id.strip():
        raise HTTPException(status_code=400, detail="institution_id cannot be empty.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: jpeg, png, webp.",
        )

    file_content = await file.read()

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Storage path uses prefixed student_id ─────────────────────────
    # student_id already arrives prefixed from Flutter e.g. NKU2400102415
    file_path = f"{student_id.strip()}/{file.filename}"

    try:
        # Remove existing file to avoid storage conflicts
        try:
            supabase.storage.from_(BUCKET).remove([file_path])
        except Exception:
            pass

        # Upload image
        supabase.storage.from_(BUCKET).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type},
        )

        image_url = supabase.storage.from_(BUCKET).get_public_url(file_path)

        # Upsert student record with institution_id
        supabase.table("students").upsert({
            "id":             student_id.strip(),
            "name":           name.strip(),
            "institution_id": institution_id.strip(),
        }).execute()

        return {
            "success":        True,
            "student_id":     student_id.strip(),
            "name":           name.strip(),
            "institution_id": institution_id.strip(),
            "image_url":      image_url,
            "file_path":      file_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/students")
def list_students(institution_id: str = None):
    """List students — optionally filtered by institution."""
    try:
        query = supabase.table("students").select("*").order("name")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        response = query.execute()
        return {"students": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    """Remove a student and their images from storage."""
    try:
        files = supabase.storage.from_(BUCKET).list(student_id)
        if files:
            paths = [f"{student_id}/{f['name']}" for f in files]
            supabase.storage.from_(BUCKET).remove(paths)
        supabase.table("students").delete().eq("id", student_id).execute()
        return {"success": True, "deleted_student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))