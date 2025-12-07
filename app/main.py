from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# Import both the user-facing and the admin-only routers
from app.routes.attendance import router as attendance_router # Handles /capture/{student_id}
from app.routes.admin import router as admin_router         # Handles /admin/attendance

load_dotenv()

app = FastAPI(title="Smart Attendance System")

# Serve static files (Assuming this is for serving the Streamlit index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/")
def home():
    # If the file exists, it returns the file; otherwise, it should ideally return a simple message
    return FileResponse(os.path.join("static", "index.html"))

# Healthcheck endpoint (Good practice for deployment monitoring)
@app.get("/health")
def health():
    return {"status": "ok"}

# 1. Include the USER/PUBLIC router (Attendance for capture/check-in)
# This router handles routes like /capture/{student_id}
# We use NO prefix here, letting the router define the full path.
app.include_router(attendance_router, tags=["Attendance"]) 

# 2. Include the SECURE ADMIN router
# This router handles routes like /admin/attendance
# We use NO prefix here, because the router file (admin.py) defines the prefix="/admin".
app.include_router(admin_router, tags=["Admin"])