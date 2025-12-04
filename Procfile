web: sh -c "streamlit run streamlit/app.py --server.port $PORT --server.address 0.0.0.0"
worker: uvicorn app.main:app --host 0.0.0.0 --port 8000