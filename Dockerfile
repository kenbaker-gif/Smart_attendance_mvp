# Use a slim Python image instead of heavy Miniconda
FROM python:3.11-slim

WORKDIR /app

# Install only basic system tools (needed for some networking/OS operations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install the 3 lightweight libraries (streamlit, supabase, python-dotenv)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Set environment variables
ENV PORT=8501
EXPOSE $PORT

# Run Streamlit directly (No more 'conda run' needed)
CMD streamlit run streamlit/app.py --server.address=0.0.0.0 --server.port=$PORT