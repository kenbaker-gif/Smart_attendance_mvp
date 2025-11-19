import sys
from pathlib import Path

# --- FIX: Robust Path Resolution ---
# 1. Resolve the path to the current directory (where app.py is: /.../smart_attendance_system/streamlit)
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Define the project root (parent of the 'streamlit' directory)
ROOT_DIR = CURRENT_DIR.parent

# 3. Insert the project root path at the beginning of sys.path
# This allows imports like 'from app.utils...' to work correctly.
sys.path.insert(0, str(ROOT_DIR))
# --- END FIX ---

import streamlit as st
import pickle
import face_recognition
import numpy as np
from PIL import Image

# This import now works because ROOT_DIR has been added to sys.path
from app.utils.encoding_utils import generate_encodings
# FIXED: Changed import back to the correct file
from app.database import add_attendance_record 

st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📸 Smart Attendance - Camera Verification")

# Robust path resolution
ENCODINGS_PATH = ROOT_DIR / "data" / "encodings_facenet.pkl"
IMAGES_DIR = ROOT_DIR / "data" / "raw_faces"

def _safe_get(data_dict, *keys):
    """Return first key that exists and is not None."""
    for k in keys:
        if k in data_dict and data_dict[k] is not None:
            return data_dict[k]
    return None

def _to_list(value):
    """Safely convert array-like to list."""
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
    """Compare dlib 128-dim encodings using face_recognition distance."""
    distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
    min_distance = np.min(distances)
    matched_index = np.argmin(distances)
    return min_distance, matched_index, min_distance < threshold

def compare_512dim_encodings(known_encodings, uploaded_encoding, threshold=0.5):
    """Compare FaceNet 512-dim encodings using Euclidean distance."""
    distances = np.linalg.norm(np.array(known_encodings) - uploaded_encoding, axis=1)
    min_distance = np.min(distances)
    matched_index = np.argmin(distances)
    return min_distance, matched_index, min_distance < threshold

# Auto-load or retrain encodings on startup
@st.cache_resource
def load_encodings():
    """Load encodings. If missing, generate them automatically."""
    if not ENCODINGS_PATH.exists():
        st.info("📂 Encodings file not found. Generating from student images...")
        # Call the imported utility function
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
        
        # Convert to list of 1D arrays
        if isinstance(encs_raw, np.ndarray):
            known_encodings = [np.array(e) for e in encs_raw]
        else:
            known_encodings = _to_list(encs_raw)
        
        known_ids = [str(i) for i in _to_list(ids_raw)]
        
        # Detect encoding dimension
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

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Capture Image")
    camera_input = st.camera_input("Point camera at your face")

with col2:
    st.subheader("ℹ️ Instructions")
    st.info("""
    1. Look directly at camera
    2. Ensure good lighting
    3. Face must be clearly visible
    4. Click "Verify" after capture
    """)

if camera_input:
    st.divider()
    
    # Convert PIL image to numpy array (PIL is RGB)
    image = Image.open(camera_input).convert("RGB")
    img_array = np.array(image)
    
    # Display captured image
    st.image(image, caption="Captured Image", width='stretch')
    
    # Try HOG first (fast)
    boxes = face_recognition.face_locations(img_array, model="hog")
    used_model = "HOG"
    
    # Fallback to CNN if HOG fails
    if len(boxes) == 0:
        with st.spinner("🔄 HOG failed, trying CNN model (slower)..."):
            try:
                boxes = face_recognition.face_locations(img_array, model="cnn")
                used_model = "CNN"
            except Exception as e:
                st.error(f"❌ CNN detection failed: {e}")
                boxes = []
    
    if len(boxes) == 0:
        st.error("❌ No face detected in image")
        st.info("💡 Try: better lighting, closer to camera, face centered")
    else:
        st.info(f"✅ Face detected using {used_model} model")
        
        # Get encodings for detected faces
        try:
            uploaded_encodings = face_recognition.face_encodings(img_array, boxes)
        except Exception as e:
            st.error(f"❌ Failed to generate encoding: {e}")
            uploaded_encodings = []
        
        if not uploaded_encodings:
            st.error("❌ Could not generate face encoding")
        else:
            st.divider()
            st.subheader("🔍 Verification Results")
            
            # Get input for verification
            student_id_input = st.text_input("Enter your Student ID to verify", placeholder="e.g., 2400102415")
            
            if st.button("✅ Verify Identity", width='stretch', type="primary"):
                if not student_id_input:
                    st.error("❌ Please enter your Student ID")
                else:
                    recognized = False
                    
                    for enc in uploaded_encodings:
                        # Check dimension match
                        if len(enc) != encoding_dim:
                            st.error(f"❌ Encoding dimension mismatch: live={len(enc)} stored={encoding_dim}")
                            break
                        
                        # Compare using appropriate method
                        if encoding_dim == 128:
                            min_distance, matched_index, is_match = compare_128dim_encodings(known_encodings, enc, threshold)
                        elif encoding_dim == 512:
                            min_distance, matched_index, is_match = compare_512dim_encodings(known_encodings, enc, threshold)
                        else:
                            st.error(f"❌ Unsupported encoding dimension: {encoding_dim}")
                            break
                        
                        matched_id = known_ids[matched_index]
                        # Confidence calculation based on distance relative to threshold
                        # Clamped between 0 and 1, assuming 0 distance is 100% confidence.
                        confidence = max(0, min(1, 1 - (min_distance / threshold)))

                        # Display match details
                        st.info(f"**Best match:** {matched_id} ({confidence:.1%} confidence)")
                        
                        # Verify if ID matches and confidence is high
                        if is_match and matched_id == student_id_input:
                            st.success(f"""
                            ✅ **VERIFICATION SUCCESSFUL**
                            
                            Student ID: {student_id_input}
                            Confidence: {confidence:.1%}
                            Distance: {min_distance:.3f}
                            """)
                            st.balloons()
                            recognized = True
                            
                            # --- DB INSERT ON SUCCESS ---
                            # FIXED: Explicitly cast confidence to standard float to avoid psycopg2 error
                            add_attendance_record(
                                student_id=student_id_input,
                                confidence=float(confidence),
                                detection_method=used_model,
                                verified="success"
                            )
                            # ---------------------------
                            
                            break
                    
                    if not recognized:
                        # --- DB INSERT ON FAILURE ---
                        # Use the best match's confidence if comparison was made, otherwise default to 0
                        final_confidence = confidence if 'confidence' in locals() else 0.0
                        
                        # FIXED: Explicitly cast confidence to standard float to avoid psycopg2 error
                        add_attendance_record(
                            student_id=student_id_input, # Still log the ID user attempted to use
                            confidence=float(final_confidence),
                            detection_method=used_model,
                            verified="failed"
                        )
                        # --------------------------
                        
                        st.error(f"""
                        ❌ **VERIFICATION FAILED**
                        
                        ID mismatch or low confidence.
                        Please try again with better lighting or different angle.
                        """)

# Admin Panel
st.divider()
with st.expander("🔧 Admin Panel"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Known Students", len(known_ids))
        st.metric("Encoding Dim", encoding_dim)
    
    with col2:
        st.metric("Threshold", f"{threshold:.2f}")
    
    if st.button("🔄 Retrain Encodings", width='stretch'):
        st.info("⏳ Retraining encodings...")
        # Call the imported utility function
        success = generate_encodings(
            images_dir=str(IMAGES_DIR),
            output_path=str(ENCODINGS_PATH)
        )
        if success:
            st.success("✅ Encodings retrained successfully!")
            st.rerun()
        else:
            st.error("❌ Retrain failed. Check data/raw_faces folder.")