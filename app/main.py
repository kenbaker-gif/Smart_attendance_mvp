# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.attendance import router as attendance_router
import os

app = FastAPI(title="Smart Attendance System")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))

# Healthcheck endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Include attendance router (Supabase upload)
app.include_router(attendance_router, prefix="/attendance", tags=["Attendance"])
