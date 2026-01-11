# 📸 Smart Attendance System (AI-Powered Face Verification)

An end-to-end AI Attendance System built with FastAPI, Streamlit, InsightFace, Supabase, and deployed on Railway.
This system performs real-time face verification pushing stored student images, automatically marks attendance, and provides a simple user interface for teachers and admins.

## 🚀 Features
## 🧠 AI Face Verification (InsightFace)

- Uses InsightFace ArcFace embeddings, one of the most accurate face verification models available.

- High-precision embedding generation and matching using cosine similarity.

- Embeddings are generated on demand (not stored in DB).

## ☁️ Cloud Storage (Supabase)

- Stores all student information and face images.

- Backend fetches images from Supabase when needed.

- Streamlined integration for reading/writing attendance records.

## ⚡ High-Performance Backend (FastAPI)

- Fast, scalable API for:

- image uploads

- embedding generation

- face matching

- attendance marking

- student management

## 🌐 Modern Frontend (Streamlit)

Clean UI for:

- capturing live webcam images

- displaying verification results

- viewing attendance logs

- managing students

## 📦 Deployment
- Backend (FastAPI)

- Deployed on Railway using:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Frontend (Streamlit)
Deployable via:
- Streamlit Cloud
- Railway
- Docker image

---

## 🚀 Quickstart
1. Clone and install dependencies
```bash
git clone <repo-url>
cd smart_attendance_system
pip install -r requirements.txt
```
2. Create a `.env` file (see **Configuration** below) and export env vars.
3. Run backend:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
4. Run frontend (Streamlit):
```bash
streamlit run streamlit/app.py
```
5. (Optional) Run with Docker:
```bash
docker build -t smart-attendance .
docker run -e PORT=8000 -p 8000:8000 smart-attendance
```

## ⚙️ Configuration
Create a `.env` file or set environment variables described below before running the app.

Example `.env`:
```
SUPABASE_URL=https://your-supabase-url.supabase.co
SUPABASE_KEY=your_service_key
PORT=8000
INSIGHTFACE_MODEL_PATH=./models/insightface.onnx
```

Recommended notes:
- Use a Supabase service-role key only when server-side operations require elevated privileges; prefer an anon key for client-side operations.
- If you precompute embeddings, document the path (e.g., `data/encodings_facenet.pkl`) and regeneration steps.

## 🏗️ System Architecture
```python

             ┌────────────────────────┐
             │        Streamlit       │
             │  (User Interface/App)  │
             └───────────┬────────────┘
                         │
                         ▼
          ┌────────────────────────────┐
          │          FASTAPI           │
          │     (Backend on Railway)   │
          └───────────┬────────────────┘
                      │
     ┌────────────────┼──────────────────┐
     │                │                  │
     ▼                ▼                  ▼
┌───────────┐   ┌────────────┐   ┌─────────────────┐
│ InsightFace│   │ Attendance │   │ Supabase Storage │
│ Embeddings │   │   Logging  │   │  (Images + Data) │
└───────────┘   └────────────┘   └─────────────────┘
```

## ⚙️ How It Works
   1. Students upload images to Supabase
      Images stored in a bucket
      Student metadata saved in Supabase DB

   2. Backend fetches images

      When attendance is triggered, the backend:
      - fetches all student images from Supabase
      - generates embeddings dynamically using InsightFace (on-demand)
      - **By default embeddings are cached in memory while the server runs** for speed
      - If you want to persist embeddings between runs, this project includes an example serialized file: `data/encodings_facenet.pkl` and Streamlit examples under `streamlit/data/encodings/` (regeneration steps may be provided in `notebooks/face_preprocessing.ipynb`).

      To regenerate encodings: fetch images, run the preprocessing notebook or script to compute and save the encodings to `data/encodings_facenet.pkl`.

   3. Live camera captures a frame
      Streamlit sends the image to FastAPI.

   4. InsightFace generates embeddings
      Both embeddings (student image + live image) are compared using cosine similarity.

   5. Attendance marked on match
      If similarity passes threshold, attendance is marked and saved in Supabase.

## 🔍 Matching Logic
- Embedding Generation
- InsightFace model → embedding vector (512-D)
- Similarity

### Cosine Similarity

The system uses **cosine similarity** to compare face embeddings generated by InsightFace:

```python
# Either L2-normalize both vectors first, or divide by product of norms:
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

## 🧪 Testing

- You can test verification by:

- Uploading student images to Supabase

- Running Streamlit

- Taking a live picture

- Watching the backend detect and mark attendance

## 🧑‍💻 Tech Stack

| Area                | Technology                          |
|:-------------------:|:-----------------------------------|
| Backend             | FastAPI, Python                     |
| AI/Face Verification | InsightFace (ArcFace), ONNX Runtime |
| Database            | Supabase (PostgreSQL + Storage)     |
| Frontend            | Streamlit                           |
| Deployment          | Railway                             |
| Auth & Management   | Supabase                            |
| Image Processing    | OpenCV, PIL                         |


## 📂 Project structure
A quick overview of the important files and folders:

- `app/` - FastAPI backend
  - `main.py` - app entrypoint
  - `routes/attendance.py` - verification and attendance endpoints
- `streamlit/` - Streamlit frontend and examples
- `utils/` - helper modules (`face_utils.py`, `supabase_utils.py`)
- `data/` - optional stored encodings and sample data

## 🛠️ Troubleshooting
- Supabase auth errors: verify `SUPABASE_URL` and `SUPABASE_KEY` environment vars.
- Model not found: set `INSIGHTFACE_MODEL_PATH` to your ONNX model and ensure the file exists.
- Camera issues (Streamlit): ensure browser permissions for camera are allowed.

## 🌍 Use Cases

- School attendance
- Employee check-in systems
- Exam hall verification
- Hostel / dormitory entry
- Visitor verification systems

## 📞 Contact / Hire Me

If you need a **custom AI attendance system** or **commercial deployment**, feel free to reach out:

**Ainebyona Abubaker**  
Freelancer | AI/ML Developer  

- Fiverr: https://www.fiverr.com/s/Ege84AD
- Email: ainebyonabubaker@proton.me

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
