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

# Install NGINX along with system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    nginx \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY . .

# Copy the Nginx configuration we created
COPY nginx.conf /etc/nginx/nginx.conf

# Create a startup script that:
# 1. Replaces PORT_PLACEHOLDER with the real Railway PORT
# 2. Starts FastAPI on 8000
# 3. Starts Streamlit on 8501
# 4. Starts Nginx to manage them both
RUN echo '#!/bin/bash\n\
sed -i "s/PORT_PLACEHOLDER/$PORT/g" /etc/nginx/nginx.conf\n\
\n\
echo "Starting FastAPI on port 8000..."\n\
uvicorn Mobile.api:app --host 0.0.0.0 --port 8000 &\n\
\n\
echo "Starting Streamlit on port 8501..."\n\
streamlit run streamlit/main.py --server.port 8501 --server.address 0.0.0.0 &\n\
\n\
echo "Starting Nginx Proxy..."\n\
nginx -g "daemon off;"\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE $PORT

CMD ["/app/start.sh"]