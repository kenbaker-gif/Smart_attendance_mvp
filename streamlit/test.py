import sys
from pathlib import Path

# --- FIX: Robust Path Resolution ---
# 1. Resolve the path to the current directory
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Define the project root (parent of the 'streamlit' directory)
ROOT_DIR = CURRENT_DIR.parent

# 3. Insert the project root path at the beginning of sys.path
sys.path.insert(0, str(ROOT_DIR))
# --- END FIX ---

import streamlit as st
import pickle
from PIL import Image
import numpy as np
import face_recognition

# NEW: Import the database function
from app.database import add_attendance_record
# Updated: Import utilities using the fixed path
from app.utils.encoding_utils import generate_encodings 


st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📸 Smart Attendance System - Image Upload")

# Robust path resolution
ENCODINGS_PATH = ROOT_DIR / "data" / "encodings_facenet.pkl"
IMAGES_DIR = ROOT_DIR / "data" / "raw_faces"

def _safe_get(data_dict, *keys):
    for k in keys:
        if k in data_dict and data_dict[k] is not None:
            return data_dict[k]
    return None

def _to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return [value]

def compare_128dim_encodings(known_encodings, uploaded_encoding, threshold=0.6):
    distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
    min_distance = np.min(distances)
    matched_index = np.argmin(distances)
    return min_distance, matched_index, min_distance < threshold

def compare_512dim_encodings(known_encodings, uploaded_encoding, threshold=0.5):
    distances = np.linalg.norm(np.array(known_encodings) - uploaded_encoding, axis=1)
    min_distance = np.min(distances)
    matched_index = np.argmin(distances)
    return min_distance, matched_index, min_distance < threshold

# Auto-load or retrain encodings on startup
@st.cache_resource
def load_encodings():
    if not ENCODINGS_PATH.exists():
        st.info("📂 Encodings file not found. Generating from student images...")
        success = generate_encodings(
            images_dir=str(IMAGES_DIR),
            output_path=str(ENCODINGS_PATH)
        )
        if not success:
            st.error("❌ Failed to generate encodings. Check data/raw_faces folder.")
            return [], [], None

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)

        encs_raw = _safe_get(data, "encodings", "encodings_facenet")
        ids_raw = _safe_get(data, "ids", "labels")

        if encs_raw is None:
            encs_raw = []
        if ids_raw is None:
            ids_raw = []

        if isinstance(encs_raw, np.ndarray):
            known_encodings = [np.array(e) for e in encs_raw]
        else:
            known_encodings = _to_list(encs_raw)

        known_ids = [str(i) for i in _to_list(ids_raw)]

        encoding_dim = len(known_encodings[0]) if known_encodings else None

        return known_encodings, known_ids, encoding_dim

    except Exception as e:
        st.error(f"❌ Failed to load encodings: {e}")
        return [], [], None

# Load encodings on startup
known_encodings, known_ids, encoding_dim = load_encodings()
threshold = 0.6 if encoding_dim == 128 else 0.5

if known_encodings:
    st.success(f"✅ Loaded {len(known_ids)} encodings ({encoding_dim}-dim)")
else:
    st.warning("⚠️ No encodings loaded. Check your setup.")

# Main UI
st.divider()
uploaded_file = st.file_uploader("Upload Student Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    student_id_input = st.text_input("Enter Student ID")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='stretch')

    if st.button("Verify Student", width='stretch'):
        if not student_id_input:
            st.error("❌ Please enter Student ID")
        elif not known_encodings:
            st.error("❌ No encodings loaded. Check encodings file.")
        else:
            img_array = np.array(image)

            # Try HOG first (fast)
            boxes = face_recognition.face_locations(img_array, model="hog")
            used_model = "hog"
            # Fallback to CNN if HOG fails
            if len(boxes) == 0:
                st.warning("⚠️ HOG model didn't detect a face. Trying CNN (slower)...")
                try:
                    boxes = face_recognition.face_locations(img_array, model="cnn")
                    used_model = "cnn"
                except Exception as e:
                    st.error(f"❌ CNN face detection failed: {e}")
                    boxes = []

            if len(boxes) == 0:
                st.error("❌ No face detected in the image")
                st.info("💡 Tips: front-facing, well-lit, unobstructed face; resolution >= 200x200")
                
                # --- DB INSERT ON FACE DETECTION FAILURE ---
                # Log the attempt, even if face wasn't found
                add_attendance_record(
                    student_id=student_id_input, 
                    confidence=0.0,
                    detection_method=used_model.upper(),
                    verified="failed"
                )
                # ------------------------------------------

            else:
                st.info(f"✅ Detected {len(boxes)} face(s) using {used_model.upper()} model")

                # Get encodings for detected faces
                try:
                    uploaded_encodings = face_recognition.face_encodings(img_array, boxes)
                except Exception as e:
                    st.error(f"❌ Failed to compute encodings: {e}")
                    uploaded_encodings = []

                if not uploaded_encodings:
                    st.error("❌ Could not generate encoding for detected face")
                    
                    # --- DB INSERT ON ENCODING FAILURE ---
                    add_attendance_record(
                        student_id=student_id_input, 
                        confidence=0.0,
                        detection_method=used_model.upper(),
                        verified="failed"
                    )
                    # ------------------------------------

                else:
                    recognized = False

                    for enc in uploaded_encodings:
                        if encoding_dim == 128:
                            min_distance, matched_index, is_match = compare_128dim_encodings(known_encodings, enc, threshold)
                        elif encoding_dim == 512:
                            min_distance, matched_index, is_match = compare_512dim_encodings(known_encodings, enc, threshold)
                        else:
                            st.error(f"❌ Unsupported encoding dimension: {encoding_dim}")
                            break

                        matched_id = known_ids[matched_index]
                        confidence = max(0, 1 - (min_distance / threshold))
                        
                        # Display best match details
                        st.info(f"**Best match:** {matched_id} ({confidence:.1%} confidence)")


                        if is_match and matched_id == student_id_input:
                            st.success(f"✅ **Student Recognized!**\n\nID: {student_id_input}\nConfidence: {confidence:.2%}")
                            st.balloons()
                            recognized = True
                            
                            # --- DB INSERT ON SUCCESS ---
                            # FIX: Cast confidence to standard float to avoid psycopg2 error
                            add_attendance_record(
                                student_id=student_id_input,
                                confidence=float(confidence),
                                detection_method=used_model.upper(),
                                verified="success"
                            )
                            # ----------------------------
                            
                            break

                    if not recognized:
                        st.warning(f"⚠️ Student not recognized or mismatch with entered ID ({student_id_input})")
                        
                        # --- DB INSERT ON FAILURE ---
                        # Use the best match's confidence if comparison was made, otherwise default to 0
                        final_confidence = confidence if 'confidence' in locals() else 0.0
                        
                        # FIX: Cast confidence to standard float to avoid psycopg2 error
                        add_attendance_record(
                            student_id=student_id_input,
                            confidence=float(final_confidence),
                            detection_method=used_model.upper(),
                            verified="failed"
                        )
                        # ----------------------------

# Admin Panel
st.divider()
st.subheader("🔧 Admin Panel")

if st.button("🔄 Retrain Encodings", width='stretch'):
    st.info("⏳ Retraining encodings... This may take a moment.")
    success = generate_encodings(
        images_dir=str(IMAGES_DIR),
        output_path=str(ENCODINGS_PATH)
    )
    if success:
        st.success("✅ Encodings retrained successfully!")
        # rerun so cached load_encodings() is refreshed on restart
        st.rerun()
    else:
        st.error("❌ Retrain failed. Check data/raw_faces folder structure.")