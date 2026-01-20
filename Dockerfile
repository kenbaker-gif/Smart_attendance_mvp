# --- STAGE 2: FINAL RUNTIME ---
FROM continuumio/miniconda3:latest
WORKDIR /app

# Essential libraries for Streamlit and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built environment from builder
COPY --from=builder /opt/conda/envs/student_env /opt/conda/envs/student_env

# Set environment variables
ENV PATH="/opt/conda/envs/student_env/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY . .

# Create the startup script to manage both processes
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI on port 8000..."\n\
uvicorn Mobile.api:app --host 0.0.0.0 --port 8000 &\n\
\n\
echo "Starting Streamlit on port $PORT..."\n\
streamlit run streamlit/main.py --server.port $PORT --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Railway uses the $PORT env var for the public web entry
EXPOSE $PORT
EXPOSE 8000

# Execute the script
CMD ["/app/start.sh"]