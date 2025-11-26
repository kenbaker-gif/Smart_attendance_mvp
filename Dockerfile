FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install precompiled dlib wheel from stable GitHub mirror
# Option 1: Use the official dlib.net link (Recommended)
RUN pip install --no-cache-dir \
    https://dlib.net/files/dlib-19.24.2-cp310-cp310-manylinux_2_28_x86_64.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]