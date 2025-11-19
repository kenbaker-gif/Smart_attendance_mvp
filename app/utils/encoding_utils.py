# app/utils/encoding_utils.py

import face_recognition
import pickle
from pathlib import Path

def generate_encodings(images_dir="data/raw_faces", output_path="data/encodings_facenet.pkl"):
    """
    Generate face encodings from image folder.

    Args:
        images_dir (str): Path to folder containing student ID subfolders with images
        output_path (str): Path to save encodings pickle file

    Returns:
        bool: True if successful, False otherwise
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    encodings = []
    ids = []

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False

    print(f"📂 Scanning {images_dir} for images...")

    for student_id_folder in sorted(images_dir.iterdir()):
        if not student_id_folder.is_dir():
            continue

        student_id = student_id_folder.name
        image_files = list(student_id_folder.glob("*.jpg")) + list(student_id_folder.glob("*.png"))

        if not image_files:
            print(f"⚠️  No images found for {student_id}")
            continue

        print(f"📸 Processing student {student_id} ({len(image_files)} images)...")

        for img_path in image_files:
            try:
                image = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(image, model="hog")

                if not face_locations:
                    print(f"  ⚠️  No face detected in {img_path.name}")
                    continue

                face_encodings = face_recognition.face_encodings(image, face_locations)

                if face_encodings:
                    encodings.append(face_encodings[0])
                    ids.append(student_id)
                    print(f"  ✅ {img_path.name}")
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")

    if encodings:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                pickle.dump({"encodings": encodings, "ids": ids}, f)
            print(f"\n✅ Saved {len(encodings)} encodings to {output_path}")
            return True
        except Exception as e:
            print(f"\n❌ Failed to save encodings: {e}")
            return False
    else:
        print("\n❌ No encodings generated. Check your image folder structure.")
        return False