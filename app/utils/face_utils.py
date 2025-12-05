# app/utils/face_utils.py
import cv2
import os
from tqdm import tqdm
from supabase import create_client
import os

# --- Supabase Config ---
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
if USE_SUPABASE:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def capture_faces(student_id, save_path="data/raw_faces"):
    """Capture images from webcam for a student"""
    os.makedirs(os.path.join(save_path, student_id), exist_ok=True)
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    print("[INFO] Starting face capture. Look at the camera...")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y + h, x:x + w]
            local_path = f"{save_path}/{student_id}/{count}.jpg"
            cv2.imwrite(local_path, face)

            # Upload to Supabase if enabled
            if USE_SUPABASE:
                with open(local_path, "rb") as f:
                    supabase.storage.from_(SUPABASE_BUCKET).upload(
                        f"{student_id}/{count}.jpg", f.read()
                    )

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Faces', frame)
        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 20:
            break

    print(f"[INFO] Captured {count} images for {student_id}")
    cam.release()
    cv2.destroyAllWindows()


def preprocess_faces(input_dir="data/raw_faces", output_dir="data/processed_faces", img_size=(160,160)):
    """Resize and normalize images for training"""
    os.makedirs(output_dir, exist_ok=True)
    students = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for student in students:
        student_input = os.path.join(input_dir, student)
        student_output = os.path.join(output_dir, student)
        os.makedirs(student_output, exist_ok=True)
        print(f"[INFO] Processing {student}...")

        for img_name in tqdm(os.listdir(student_input), desc=student):
            input_path = os.path.join(student_input, img_name)
            output_path = os.path.join(student_output, img_name)
            try:
                img = cv2.imread(input_path)
                if img is None:
                    continue
                resized = cv2.resize(img, img_size)
                normalized = resized / 255.0
                cv2.imwrite(output_path, (normalized * 255).astype("uint8"))
            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {e}")

    print(f"[DONE] Preprocessing completed. Processed faces saved to {output_dir}")
