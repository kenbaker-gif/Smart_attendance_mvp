from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Student(Base):
    """Student information."""
    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Student(id={self.id}, name={self.name})>"


class AttendanceRecord(Base):
    """Attendance verification records."""
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, ForeignKey("students.id"), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float)  # 0.0 to 1.0
    detection_method = Column(String)  # "hog" or "cnn"
    verified = Column(String)  # "success" or "failed"

    def __repr__(self):
        return f"<AttendanceRecord(student_id={self.student_id}, confidence={self.confidence})>"