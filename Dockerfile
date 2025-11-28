# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libx11-6 \
    libsm6 \
    libxrender1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install heavy ML dependencies in conda
RUN conda install -y tensorflow==2.13 opencv && conda clean -afy

# Install pip deps (deepface, streamlit, supabase)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install deepface

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Reinstall minimal run dependencies (optional but safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libx11-6 \
    libsm6 \
    libxrender1 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy conda env
COPY --from=builder /opt/conda /opt/conda

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy application code
COPY . .

# Streamlit port
ENV PORT=8501
EXPOSE $PORT

CMD ["sh", "-c", "streamlit run streamlit/app.py --server.address=0.0.0.0 --server.port=$PORT"]