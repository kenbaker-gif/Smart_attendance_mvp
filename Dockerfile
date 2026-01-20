# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN conda create -n student_env python=3.11 -y && \
    conda install -n student_env -c conda-forge opencv onnxruntime -y --quiet && \
    /opt/conda/envs/student_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# --- STAGE 2: FINAL RUNTIME ---
FROM continuumio/miniconda3:latest
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY . .

# Re-create the startup script to ensure it's fresh
RUN echo '#!/bin/bash\n\
uvicorn Mobile.api:app --host 0.0.0.0 --port 8000 &\n\
streamlit run streamlit/main.py --server.port $PORT --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE $PORT
EXPOSE 8000

CMD ["/app/start.sh"]