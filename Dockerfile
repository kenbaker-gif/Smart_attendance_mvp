--- STAGE 1: BUILDER (The Heavy Lifting) ---

We use the miniconda image to successfully install dlib and opencv via conda.

FROM continuumio/miniconda3:latest AS builder

1. Install minimal system dependencies required for the base environment and eventual runtime

RUN apt-get update && apt-get install -y 

libgl1 

libopenblas-dev 

liblapack-dev 

&& rm -rf /var/lib/apt/lists/*

2. Set environment and copy requirements

WORKDIR /app
COPY requirements.txt .

3. Install necessary conda packages (dlib and OpenCV)

We use dlib=19.24 as you specified.

RUN conda install -c conda-forge dlib=19.24 opencv -y --no-update-deps --quiet

4. Install remaining pip packages into the conda environment

This assumes requirements.txt does NOT contain dlib or opencv

RUN pip install --no-cache-dir -r requirements.txt

--- STAGE 2: FINAL (The Minimal Runtime) ---

Start fresh with a slim Python base image. This keeps the final image small.

FROM python:3.10-slim

1. Install minimal runtime dependencies needed by dlib/OpenCV

These packages are still needed for the binaries to link correctly at runtime.

RUN apt-get update && apt-get install -y --no-install-recommends 

libgl1 

libopenblas0 

liblapack3 

&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

2. Copy ONLY the finalized Conda environment from the builder stage

The 'opt/conda' directory contains all the installed binaries and Python site-packages.

COPY --from=builder /opt/conda /opt/conda

3. Set the PATH to include the Conda binaries (IMPORTANT!)

ENV PATH="/opt/conda/bin:$PATH"

4. Copy application code

COPY . .

Final configuration

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]