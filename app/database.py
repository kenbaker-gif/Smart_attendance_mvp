from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from app.models import Base, Student, AttendanceRecord
from datetime import date, datetime
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus, parse_qs, urlencode # Added imports for robust option handling

# --- CONFIGURATION ---

# Load variables from the project root's .env file
load_dotenv() 

# Helper function to load and clean environment variables
def get_cleaned_env_var(key, clean_quotes=False):
    value = os.getenv(key)
    if value is None:
        return None
    
    # Strip leading/trailing whitespace
    cleaned_value = value.strip()
    
    # Aggressively remove common quote characters and backticks if cleaning credentials
    if clean_quotes:
        # Note: We are removing quotes/backticks here, but we rely on quote_plus later for encoding special characters
        cleaned_value = cleaned_value.replace('"', '').replace("'", "").replace("`", "")

    return cleaned_value

# Access individual database components and remove leading/trailing whitespace using .strip()
# Use aggressive cleaning for sensitive variables (USER, PASSWORD)
DB_USER = get_cleaned_env_var("DB_USER", clean_quotes=True)
DB_PASSWORD = get_cleaned_env_var("DB_PASSWORD", clean_quotes=True)
DB_HOST = get_cleaned_env_var("DB_HOST")
DB_PORT = get_cleaned_env_var("DB_PORT") # This port is mostly ignored for Pooler
DB_NAME = get_cleaned_env_var("DB_NAME")
DB_OPTIONS_RAW = get_cleaned_env_var("DB_OPTIONS") or "" # Renamed to DB_OPTIONS_RAW

# CRUCIAL: Check if all necessary variables were loaded successfully
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise EnvironmentError(
        "One or more required database environment variables (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME) "
        "not found. Please check your .env file."
    )

# Use the recommended Supabase Connection Pooler port (6543)
SUPABASE_POOLER_PORT = "6543" 

# Properly URL-encode the password to handle special characters (CRITICAL FIX)
ENCODED_DB_PASSWORD = quote_plus(DB_PASSWORD)

# --- SASL FIX: Ensure client_encoding=utf8 is present in options ---
# This often fixes the 'duplicate SASL authentication request' error with poolers.

# 1. Strip the leading '?' if present from DB_OPTIONS_RAW
options_string = DB_OPTIONS_RAW.lstrip('?')

# 2. Parse the options into a dictionary
options_dict = parse_qs(options_string)

# 3. Apply necessary fixes: SASL fix
# Ensure client_encoding=utf8 is set
if 'client_encoding' not in options_dict:
    options_dict['client_encoding'] = ['utf8']
    
# 4. Rebuild the options string
DB_OPTIONS_ENCODED = f"?{urlencode(options_dict, doseq=True)}" if options_dict else ""
# --- END SASL FIX ---


# Construct the SQLAlchemy Database URL
# NOTE: The DB_HOST must now be the Supavisor pooler hostname (e.g., aws-0-us-east-1.pooler.supabase.com)
SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{DB_USER}:{ENCODED_DB_PASSWORD}@{DB_HOST}:{SUPABASE_POOLER_PORT}/{DB_NAME}{DB_OPTIONS_ENCODED}"
)

# --- DEBUGGING STEP: PRINT THE CONSTRUCTED URL (Excluding password for security) ---
print("--- DATABASE CONNECTION INFO ---")
print(f"Host: {DB_HOST} (Expected: Supavisor Pooler Hostname)")
print(f"Port: {SUPABASE_POOLER_PORT}")
print(f"URL (masked): postgresql://{DB_USER}:***@{DB_HOST}:{SUPABASE_POOLER_PORT}/{DB_NAME}{DB_OPTIONS_ENCODED}")
print("--------------------------------")


# Now create the engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)

# Initialize SessionLocal
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables (This will only create tables that don't already exist)
Base.metadata.create_all(bind=engine)

# --- DEPENDENCY AND HELPER FUNCTIONS ---

# Dependency for FastAPI
def get_db():
    """Provides a fresh database session for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_attendance_record(student_id: str, confidence: float, detection_method: str, verified: str):
    """Add an attendance record to the database."""
    db = SessionLocal()
    try:
        record = AttendanceRecord(
            student_id=student_id,
            confidence=confidence,
            detection_method=detection_method,
            verified=verified
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as e:
        db.rollback()
        print(f"Error adding attendance record: {e}")
        return None
    finally:
        db.close()

def get_student_attendance(student_id: str, limit: int = 10):
    """Get recent attendance records for a student."""
    db = SessionLocal()
    try:
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.student_id == student_id
        ).order_by(AttendanceRecord.timestamp.desc()).limit(limit).all()
        return records
    finally:
        db.close()

def get_today_attendance():
    """Get today's attendance records."""
    # The 'datetime' import is now correctly at the top of the file.
    db = SessionLocal()
    try:
        today = date.today()
        # Use datetime.combine() with min.time() to get the start of the day
        start_of_day = datetime.combine(today, datetime.min.time()) 
        
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.timestamp >= start_of_day
        ).order_by(AttendanceRecord.timestamp.desc()).all()
        return records
    finally:
        db.close()

def get_attendance_summary(student_id: str = None):
    """Get attendance summary statistics."""
    db = SessionLocal()
    try:
        # Query for total verified records
        query = db.query(AttendanceRecord).filter(
            AttendanceRecord.verified == "success"
        )
        if student_id:
            query = query.filter(AttendanceRecord.student_id == student_id)

        total = query.count()

        # Query for average confidence
        avg_conf_query = db.query(func.avg(AttendanceRecord.confidence)).filter(
            AttendanceRecord.verified == "success"
        )
        if student_id:
            avg_conf_query = avg_conf_query.filter(AttendanceRecord.student_id == student_id)

        avg_confidence = avg_conf_query.scalar()

        return {
            "total_verified": total,
            "avg_confidence": float(avg_confidence) if avg_confidence else 0
        }
    finally:
        db.close()

def register_student(student_id: str, name: str = None, email: str = None):
    """Register a new student."""
    db = SessionLocal()
    try:
        # Check if student already exists
        existing = db.query(Student).filter(Student.id == student_id).first()
        if existing:
            return existing

        # Create new student record
        student = Student(id=student_id, name=name, email=email)
        db.add(student)
        db.commit()
        db.refresh(student)
        return student
    except Exception as e:
        db.rollback()
        print(f"Error registering student: {e}")
        return None
    finally:
        db.close()

def get_all_students():
    """Get all registered student records."""
    db = SessionLocal()
    try:
        # Retrieve all student records from the Student table
        students = db.query(Student).all()
        return students
    finally:
        db.close()

def get_student_by_id(student_id: str):
    """Get a student record by their ID."""
    db = SessionLocal()
    try:
        # Filter the Student table by the provided student ID
        student = db.query(Student).filter(Student.id == student_id).first()
        return student
    finally:
        db.close()