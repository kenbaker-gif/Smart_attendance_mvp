# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

# System dependencies for compiling InsightFace/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create env and install heavy binaries via Conda
RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# --- STAGE 2: FINAL RUNTIME ---
FROM continuumio/miniconda3:latest
WORKDIR /app

# Essential libraries for Streamlit and OpenCV to run in Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built environment
COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# Set environment variables
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PORT=8501

COPY . .

# Expose Streamlit's default port
EXPOSE $PORT

# Run Streamlit using the Conda environment
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]