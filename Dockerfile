# =============================================================================
# STAGE 1: BUILDER
# =============================================================================
# We use a full miniconda image here to compile heavy packages like InsightFace
# and OpenCV. The compiled environment is then copied into a clean runtime
# image, keeping the final image lean.
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

# Install C/C++ build tools needed to compile InsightFace, OpenCV, and other
# packages that have native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first. Docker caches this layer — if requirements.txt
# hasn't changed, the expensive conda/pip install below is skipped on rebuild.
COPY requirements.txt .

# Build the Python environment.
# - opencv and onnxruntime come from conda-forge for better binary compatibility.
# - Everything else comes from pip.
# - conda clean removes package cache to reduce layer size.
RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy


# =============================================================================
# STAGE 2: RUNTIME
# =============================================================================
FROM continuumio/miniconda3:latest
WORKDIR /app

# Minimal system libraries required at runtime by OpenCV and InsightFace.
# libgl1        → OpenCV needs libGL.so.1 for image processing
# libglib2.0-0  → OpenCV needs libgobject / glib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the fully-built conda environment from the builder stage.
# Nothing else from the builder (build tools, cache) comes with it.
COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# FIX 3 (OpenBLAS thread explosion):
# OpenBLAS (used internally by numpy) defaults to spawning 32 threads per
# process. With 4 uvicorn workers that's 4 × 32 = 128 threads all trying to
# spawn at startup, which exhausts Railway's container thread limit and kills
# workers with: pthread_create failed ... Resource temporarily unavailable
#
# Setting this to 1 means each worker uses a single OpenBLAS thread.
# There is no meaningful performance loss — the real bottleneck in FaceAttend
# is ONNX/InsightFace inference, not numpy cosine similarity math.
ENV OPENBLAS_NUM_THREADS=1

# Same fix applied to other common BLAS/threading backends in case the
# conda environment resolves to one of these instead of OpenBLAS.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Default port for local testing. Railway overrides this at runtime.
ENV PORT=8000

# Create the data directory your app expects for encodings etc.
RUN mkdir -p /app/app/streamlit/data

# Copy source code last — it changes most often, so keeping it at the bottom
# means Docker can reuse all the expensive layers above on most rebuilds.
COPY . .

# -----------------------------------------------------------------------------
# FIX 1 (Container-level race condition — InsightFace buffalo_s model):
# Pre-download the InsightFace buffalo_s weights at IMAGE BUILD TIME.
#
# WHY THIS IS NEEDED:
# uvicorn spawns N worker processes, and each calls preload_models() almost
# simultaneously at startup. Without this, every worker races to:
#   1. Check if /root/.insightface/models/buffalo_s exists  → it doesn't
#   2. Call os.makedirs() to create it
#   3. Download buffalo_s.zip from GitHub
#
# The first worker to finish makedirs() wins. All others crash with:
#   FileExistsError: [Errno 17] File exists: '.../buffalo_s'
#
# By downloading during docker build, the directory and weights are already
# on disk when the container starts. Every worker finds them immediately —
# no download, no race, no crash. Cold starts are also faster as a bonus.
# -----------------------------------------------------------------------------
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis( \
    name='buffalo_s', \
    root='/root/.insightface', \
    allowed_modules=['detection', 'recognition'] \
); \
print('✅ buffalo_s model pre-downloaded successfully.')"

# Pre-download the uniface MiniFASNetV2 anti-spoofing weights at build time.
# Same reasoning as above — avoids all workers racing to download at startup.
RUN python3 -c "\
from uniface import create_spoofer; \
create_spoofer(); \
print('✅ Anti-spoof model pre-downloaded successfully.')"

EXPOSE 8000

# -----------------------------------------------------------------------------
# Worker count: 4 (not 8)
#
# Railway's "8 vCPUs" are shared/burstable, not dedicated cores.
# Running 8 workers means:
#   - 8 full copies of InsightFace + anti-spoof models in RAM simultaneously
#   - 8 × ONNX Runtime thread pools all competing for CPU
#
# 4 workers is the right balance for a university attendance workload:
#   - Still handles solid concurrent request throughput
#   - Halves memory pressure
#   - Leaves headroom so workers don't starve each other during face inference
# -----------------------------------------------------------------------------
CMD uvicorn Mobile.api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 4 \
    --timeout-keep-alive 60