# ~/Smart_attendance_mvp/dispatch.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from Mobile.api import app as api_app
from streamlit.web.cli import main as st_cli
import os

# This merges them onto one port
# Your phone hits: https://your-url.up.railway.app/verify
# Your browser hits: https://your-url.up.railway.app/