# --- STAGE 1: BUILDER (Standard Setup) ---
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create Conda environment
RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# --- STAGE 2: FINAL RUNTIME (API Only) ---
FROM continuumio/miniconda3:latest
WORKDIR /app

# Install only the system libraries needed for OpenCV (No NGINX)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the prepared environment
COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# Set Path
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create data directory (Good practice to keep)
RUN mkdir -p /app/app/streamlit/data

# Copy your code
COPY . .

# Run the Mobile API directly
# This uses the $PORT variable from Railway automatically
CMD uvicorn Mobile.api:app --host 0.0.0.0 --port $PORT