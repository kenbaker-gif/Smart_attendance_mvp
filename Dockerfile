# Use a slim Python image instead of heavy Miniconda
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV/InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
EXPOSE $PORT

# Run FastAPI with Uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT