# --- STAGE 1: BASE IMAGE ---
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libx11-6 \
    libsm6 \
    libxrender1 \
    libopenblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install InsightFace
RUN pip install --no-cache-dir insightface[onnx]

# --- STAGE 2: FINAL IMAGE ---
FROM python:3.11-slim

WORKDIR /app

# System dependencies (runtime only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libx11-6 \
    libsm6 \
    libxrender1 \
    libopenblas0 \
    liblapack3 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin

# Copy project code
COPY . .

# Pre-download/copy InsightFace model to cache
ENV INSIGHTFACE_CACHE_DIR=/app/models_cache
RUN mkdir -p $INSIGHTFACE_CACHE_DIR
ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0
ENV MXNET_CPU_WORKER_NTHREADS=1
ENV MXNET_GPU_WORKER_NTHREADS=1
ENV INSIGHTFACE_ROOT=$INSIGHTFACE_CACHE_DIR

# Streamlit & FastAPI ports
ENV STREAMLIT_PORT=8501
ENV FASTAPI_PORT=8000
EXPOSE $STREAMLIT_PORT
EXPOSE $FASTAPI_PORT

# Start both FastAPI (Uvicorn) and Streamlit concurrently
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $FASTAPI_PORT & streamlit run streamlit/app.py --server.address=0.0.0.0 --server.port=$STREAMLIT_PORT"]
