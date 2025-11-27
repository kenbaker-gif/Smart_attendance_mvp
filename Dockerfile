# --- STAGE 1: BUILDER (For Conda ML Dependencies) ---
FROM continuumio/miniconda3:latest AS builder

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    # The 'git' and 'cmake' packages are often needed to build dlib from scratch,
    # though conda handles the actual installation here.
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements (assuming it includes streamlit, numpy, face-recognition, etc.)
COPY requirements.txt .

# Install heavy dependencies (dlib, opencv) using conda
# Use 'conda install' before 'pip install' to resolve conflicts better
RUN conda install -c conda-forge dlib=19.24 opencv -y --no-update-deps --quiet

# Install remaining Python packages (must include 'streamlit' and 'supabase')
RUN pip install --no-cache-dir -r requirements.txt

# Clean conda cache to reduce size
RUN conda clean -afy

# --- STAGE 2: FINAL (Minimal Runtime) ---
# Use a slim base image
FROM python:3.10-slim

# Minimal runtime dependencies for the ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy prebuilt conda environment from the builder stage
COPY --from=builder /opt/conda /opt/conda

ENV PATH="/opt/conda/bin:$PATH"

# Copy the application code (app.py, recognition files, etc.)
COPY . .

# Set the default port for Streamlit
ENV PORT=8501

# Expose Streamlit's default port
EXPOSE $PORT

# --- CRITICAL CHANGE: Use the Streamlit command ---
# CMD to run your Streamlit application script
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]