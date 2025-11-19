from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# --- Student Schemas ---

class StudentCreate(BaseModel):
    """Schema for creating a new student (input validation)."""
    # ID is required for registration
    id: str = Field(..., example="stu123", description="Unique ID for the student.")
    name: Optional[str] = Field(None, example="Jane Doe")
    email: Optional[str] = Field(None, example="jane.doe@example.com")

class StudentOut(BaseModel):
    """Schema for returning full student data (output formatting)."""
    id: str
    name: Optional[str]
    email: Optional[str]
    created_at: datetime # Include the creation timestamp

    class Config:
        # Allows conversion from SQLAlchemy ORM models to Pydantic objects
        orm_mode = True 

# --- Attendance Schemas ---

class AttendanceRecordIn(BaseModel):
    """Schema for creating a new attendance record (input validation)."""
    student_id: str = Field(..., example="stu123", description="ID of the student detected.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the facial recognition.")
    detection_method: str = Field(..., example="cnn", description="Method used for detection (e.g., 'hog', 'cnn').")
    # 'verified' status might be set by the system logic, but we include it here for manual input options
    verified: str = Field(..., example="success", description="Verification status ('success' or 'failed').")


class AttendanceRecordOut(BaseModel):
    """Schema for returning attendance record data (output formatting)."""
    id: int # Include the record ID
    student_id: str
    timestamp: datetime
    confidence: float
    detection_method: str
    verified: str

    class Config:
        orm_mode = True
        
# --- Utility Schemas ---

class Message(BaseModel):
    """Generic message schema for sending simple status responses."""
    message: str