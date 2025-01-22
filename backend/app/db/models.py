from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# models.py
class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    video_name = Column(String)
    video_url = Column(String, nullable=True)
    source_type = Column(String)
    detection_time = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float, default=0.0)
    total_frames = Column(Integer, default=0)
    storage_path = Column(String)

class DetectionResult(Base):
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    brand = Column(String)
    confidence = Column(Float)
    frame_number = Column(Integer, nullable=True)
    time_stamp = Column(Float, nullable=True)
    bbox_coordinates = Column(String)  # JSON string: [x1, y1, x2, y2]
    bbox_image_path = Column(String)
    screen_time_percentage = Column(Float, nullable=True)

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    content = Column(Text)
    summary_stats = Column(Text)  # JSON string con estadísticas
    created_at = Column(DateTime(timezone=True), server_default=func.now())