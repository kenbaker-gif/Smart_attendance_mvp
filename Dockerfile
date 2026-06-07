# =============================================================================
# STAGE 1: BUILDER
# =============================================================================
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy


# =============================================================================
# STAGE 2: RUNTIME
# =============================================================================
FROM continuumio/miniconda3:latest
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# =============================================================================
# Environment variables
# =============================================================================
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV ORT_NUM_THREADS=1

# =============================================================================
# Filesystem setup
# =============================================================================
RUN mkdir -p /app/app/streamlit/data

COPY . .

# =============================================================================
# Pre-download model weights at IMAGE BUILD TIME
# =============================================================================
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis( \
    name='buffalo_s', \
    root='/root/.insightface', \
    allowed_modules=['detection', 'recognition'] \
); \
print('✅ buffalo_s pre-downloaded successfully.')"

RUN python3 -c "\
from uniface import create_spoofer; \
create_spoofer(); \
print('✅ Anti-spoof model pre-downloaded successfully.')"

EXPOSE 8000

CMD uvicorn Mobile.api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 2 \
    --timeout-keep-alive 60