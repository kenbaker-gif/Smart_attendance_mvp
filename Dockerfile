# =============================================================================
# STAGE 1: BUILDER
# =============================================================================
# Full miniconda image for compiling heavy native packages (InsightFace,
# OpenCV). The compiled environment is copied into a clean runtime image,
# keeping the final image lean and free of build tools.
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

# C/C++ build tools required to compile InsightFace, OpenCV, and other
# packages that have native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first. Docker caches this layer — if requirements.txt
# hasn't changed, the expensive conda/pip install below is skipped on rebuild.
COPY requirements.txt .

# Build the Python 3.11 environment.
# - opencv + onnxruntime come from conda-forge for better binary compatibility.
# - Everything else from pip.
# - conda clean removes package cache to keep the layer small.
RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy


# =============================================================================
# STAGE 2: RUNTIME
# =============================================================================
FROM continuumio/miniconda3:latest
WORKDIR /app

# Minimal runtime libraries required by OpenCV and InsightFace.
# libgl1        → OpenCV needs libGL.so.1 for image processing
# libglib2.0-0  → OpenCV needs libgobject/glib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the built environment from builder — no build tools follow.
COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# =============================================================================
# Environment variables
# =============================================================================
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default port for local testing. Railway overrides this at runtime.
ENV PORT=8000

# -----------------------------------------------------------------------------
# FIX 3a: Cap OpenBLAS threads (numpy backend).
#
# OpenBLAS defaults to 32 threads per process. With 4 workers that's
# 4 × 32 = 128 threads fighting over 8 CPUs at startup, causing:
#   pthread_create failed ... Resource temporarily unavailable
# 1 thread per worker is enough — numpy cosine similarity on 17–1000
# embeddings is not the bottleneck, ONNX inference is.
# -----------------------------------------------------------------------------
ENV OPENBLAS_NUM_THREADS=1

# Same cap applied to other BLAS backends in case conda resolves to one
# of these instead of OpenBLAS (MKL on Intel, generic OpenMP).
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# -----------------------------------------------------------------------------
# FIX 3b: Cap ONNX Runtime internal thread pool per worker.
#
# ONNX Runtime has its own thread pool separate from OpenBLAS/OMP.
# Without this cap it tries to claim all 8 CPUs per worker process:
#   4 workers × 8 ONNX threads = 32 threads competing for 8 real cores
# This causes workers to starve each other, producing the staggered
# latency seen in stress tests (Thread 0: 4.59s → Thread 1: 7.31s → 9.41s).
#
# With ORT_NUM_THREADS=2 and 4 workers:
#   4 workers × 2 ONNX threads = 8 threads = exactly 8 vCPUs
# Each worker gets clean dedicated CPU time. No contention.
# Expected: ~0.15–0.20s single request, ~0.3–0.5s at 5 concurrent.
# -----------------------------------------------------------------------------
ENV ORT_NUM_THREADS=2

# =============================================================================
# Filesystem setup
# =============================================================================
RUN mkdir -p /app/app/streamlit/data

# Copy source code last — changes most often, so Docker can reuse all the
# expensive layers above on most rebuilds.
COPY . .

# =============================================================================
# FIX 1: Pre-download model weights at IMAGE BUILD TIME.
# =============================================================================
# WHY: uvicorn spawns 4 worker processes and each calls preload_models()
# almost simultaneously at startup. Without pre-downloading, every worker
# races to:
#   1. Check if /root/.insightface/models/buffalo_s exists  → it doesn't
#   2. Call os.makedirs() to create it
#   3. Download buffalo_s.zip from GitHub
#
# The first worker to finish makedirs() wins. All others crash with:
#   FileExistsError: [Errno 17] File exists: '.../buffalo_s'
#
# Downloading at build time means the directory and weights are already on
# disk when the container starts. All 4 workers find them immediately —
# no download, no race, no crash. Cold starts are faster too.

# Pre-download InsightFace buffalo_s weights.
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis( \
    name='buffalo_s', \
    root='/root/.insightface', \
    allowed_modules=['detection', 'recognition'] \
); \
print('✅ buffalo_s pre-downloaded successfully.')"

# Pre-download uniface MiniFASNetV2 anti-spoofing weights.
# Same reasoning — avoids all workers racing to download at startup.
RUN python3 -c "\
from uniface import create_spoofer; \
create_spoofer(); \
print('✅ Anti-spoof model pre-downloaded successfully.')"

EXPOSE 8000

# =============================================================================
# Worker configuration: 4 workers on 8 vCPUs
# =============================================================================
# With ORT_NUM_THREADS=2, each worker uses exactly 2 CPU cores for ONNX
# inference: 4 × 2 = 8 cores = full utilisation of available hardware.
#
# Why not 8 workers?
# - 8 workers × 2 ONNX threads = 16 threads on 8 cores → contention returns
# - 8 full model copies in RAM vs 4 (InsightFace + antispoof ~400MB each)
# - 4 workers is the sweet spot for this hardware and workload
#
# --timeout-keep-alive 60: holds connections open for 60s to avoid
# reconnect overhead from the Flutter app between rapid scan requests.
# =============================================================================
CMD uvicorn Mobile.api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 4 \
    --timeout-keep-alive 60