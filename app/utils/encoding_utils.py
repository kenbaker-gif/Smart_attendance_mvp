import pickle
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

# -----------------------------
# CONSTANTS & TYPES
# -----------------------------
# Defines the expected data structure for our encodings file
EncodingsData = Dict[str, np.ndarray]

# -----------------------------
# ENCODING UTILITIES
# -----------------------------

def load_encodings(path: Path) -> Optional[EncodingsData]:
    """
    Loads face encodings (embeddings) and corresponding IDs from a pickle file.
    
    The expected format is a dictionary:
    { "encodings": numpy.ndarray (N, 512), "ids": numpy.ndarray (N,) }
    
    Returns the dictionary data on success, or None on failure/file not found.
    """
    if not path.exists():
        print(f"⚠ Encodings file not found at: {path}")
        return None
    
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        
        if isinstance(data, dict) and "encodings" in data and "ids" in data:
            print(f"✅ Loaded {len(data['encodings'])} encodings from {path.name}.")
            # Ensure they are numpy arrays of the correct type
            data["encodings"] = np.array(data["encodings"], dtype=np.float32)
            data["ids"] = np.array(data["ids"], dtype=str)
            return data
        else:
            print(f"❌ Encodings file format is invalid: {path}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load encodings from {path}: {e}")
        return None

def save_encodings(data: EncodingsData, path: Path) -> bool:
    """
    Saves face encodings and corresponding IDs to a pickle file.
    
    Args:
        data: A dictionary containing "encodings" (Numpy array of embeddings)
              and "ids" (Numpy array of corresponding IDs).
        path: The Path object where the pickle file should be saved.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
            
        print(f"✅ Successfully saved {len(data['encodings'])} encodings to {path}.")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save encodings to {path}: {e}")
        return False