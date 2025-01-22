from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# User schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: str | None = None
    
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Detection schemas
class DetectionBase(BaseModel):
    video_name: str
    video_url: Optional[str] = None
    source_type: str
    storage_path: str
    duration_seconds: Optional[float] = Field(default=0.0)
    total_frames: Optional[int] = Field(default=0)

class DetectionCreate(DetectionBase):
    user_id: int

class Detection(DetectionBase):
    id: int
    user_id: int
    detection_time: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

# DetectionResult schemas
class DetectionResultBase(BaseModel):
    brand: str
    confidence: float
    frame_number: Optional[int] = None
    time_stamp: Optional[float] = None
    bbox_coordinates: Optional[str] = None
    screen_time_percentage: Optional[float] = None

class DetectionResultCreate(DetectionResultBase):
    detection_id: int

class DetectionResult(DetectionResultBase):
    id: int
    detection_id: int
    bbox_image_path: Optional[str] = None

    class Config:
        from_attributes = True

# Report schemas
class ReportBase(BaseModel):
    content: str
    summary_stats: str

class ReportCreate(ReportBase):
    detection_id: int

class Report(ReportBase):
    id: int
    detection_id: int
    created_at: datetime

    class Config:
        from_attributes = True