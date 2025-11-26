FROM python:3.10-slim

# Minimal system deps required by face_recognition / OpenCV runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install precompiled dlib wheel (Python 3.10, manylinux_2_28 x86_64)
RUN pip install --no-cache-dir \
    https://dlib.net/files/dlib-19.24.2-cp310-cp310-manylinux_2_28_x86_64.whl

# Install remaining Python deps (ensure dlib is NOT listed in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
