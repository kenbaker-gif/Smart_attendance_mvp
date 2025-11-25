# Use a Miniconda base image
FROM continuumio/miniconda3

# --- 1. INSTALL MISSING SYSTEM LIBRARIES ---
# The 'libGL.so.1' error requires this package, even when using Conda.
RUN apt-get update && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# 2. Install dlib AND opencv using Conda (FAST and robust for binary dependencies)
RUN conda install -c conda-forge dlib opencv -y

# 3. Install Python dependencies via pip
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application files and necessary configuration
COPY .env /app/.env
COPY . /app

# The EXPOSE and CMD lines are handled by docker-compose.yml