# --- STAGE 1: BUILDER ---
# Using miniconda to build the heavy environment
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

# Install build tools for packages like InsightFace/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Create the environment and install dependencies
# Note: student_env is the name you used previously
RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# --- STAGE 2: FINAL RUNTIME ---
FROM continuumio/miniconda3:latest
WORKDIR /app

# Install system-level dependencies for OpenCV/InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the built environment from the builder stage
COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# Set environment variables
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1
# Default port for local testing, Railway will override this
ENV PORT=8000

# Create necessary directories for your smart attendance system
RUN mkdir -p /app/app/streamlit/data

# Copy your source code last (since it changes most often)
COPY . .

# FIX 1 (Container-level race condition): Pre-download the InsightFace buffalo_s
# model weights at IMAGE BUILD TIME, not at container startup.
#
# WHY THIS IS NEEDED:
# Railway runs 8 worker processes (--workers 8). When the container starts,
# all 8 workers call preload_models() almost simultaneously. Each one checks
# if /root/.insightface/models/buffalo_s exists, sees it doesn't, and races
# to create it and download the zip. The first one wins; the other 7 crash
# with: FileExistsError: [Errno 17] File exists: '.../buffalo_s'
#
# By downloading during docker build, the directory already exists on disk
# when the container starts. All 8 workers find it immediately — no download,
# no race, no crash. This also makes cold starts faster.
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_s', root='/root/.insightface', allowed_modules=['detection', 'recognition']); \
print('buffalo_s model pre-downloaded successfully.')"

# Pre-download uniface antispoof model weights at build time
# (Same reasoning: avoid all workers racing to download this at startup)
RUN python3 -c "from uniface import create_spoofer; create_spoofer()"

# EXPOSE is optional for Railway but good for documentation
EXPOSE 8000

# THE FIX: 8 Workers for 8vCPUs to maximize parallel processing
CMD uvicorn Mobile.api:app --host 0.0.0.0 --port $PORT --workers 8 --timeout-keep-alive 60