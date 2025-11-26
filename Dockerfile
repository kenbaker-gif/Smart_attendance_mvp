FROM continuumio/miniconda3:latest

WORKDIR /app

# Minimal system packages only
RUN apt-get update && apt-get install -y \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install only necessary conda packages
RUN conda install -c conda-forge dlib=19.24 opencv -y \
    --no-update-deps \
    --quiet

# Install Python packages (without dlib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
