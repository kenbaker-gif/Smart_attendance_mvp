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

# EXPOSE is optional for Railway but good for documentation
EXPOSE 8000

# THE FIX: Using "Shell Form" (no brackets) so $PORT expands correctly
CMD uvicorn Mobile.api:app --host 0.0.0.0 --port $PORT