# --- STAGE 1: BUILDER (Heavy Install & Cleaning) ---
# We use this stage to perform all installations and clean up the cache.
FROM continuumio/miniconda3:latest AS final

# Install minimal system dependencies needed for runtime
# We must do this *after* the FROM command for the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libopenblas0 \
    liblapack3 \
    git \
    cmake \
    # Clean up the apt lists immediately after install
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# 1. Install heavy dependencies (dlib, opencv) using conda
RUN conda install -c conda-forge dlib=19.24 opencv -y --no-update-deps --quiet

# 2. Install remaining Python packages (must include 'streamlit' and 'supabase')
RUN pip install --no-cache-dir -r requirements.txt

# 3. Clean conda cache to reduce image size (CRITICAL STEP)
# This significantly shrinks the final image before saving it.
RUN conda clean -afy

# --- STAGE 2: APPLICATION RUNTIME (NO SEPARATE STAGE NEEDED) ---
# Since Stage 1 is the most efficient final image, we rename it 'final'
# and continue. The environment is already configured.

# Copy the application code (app.py, recognition files, etc.)
# Do this last so code changes don't invalidate the slow dependency layer cache.
COPY . .

# Set the default port for Streamlit
ENV PORT=8501

# Expose Streamlit's default port
EXPOSE $PORT

# Set the PATH environment variable (already configured in the base image, but good practice)
ENV PATH="/opt/conda/bin:$PATH"

# CMD to run your Streamlit application script
CMD ["sh", "-c", "streamlit run streamlit/app.py --server.address=0.0.0.0 --server.port=$PORT"]