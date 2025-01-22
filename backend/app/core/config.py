from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path
import os
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent)
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY")
    ROBOFLOW_API_KEY: Optional[str] = os.getenv("ROBOFLOW_API_KEY")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "tu_clave_secreta_por_defecto")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
    
    # Storage paths
    STORAGE_PATH: Path = Field(default_factory=lambda: Path("storage"))
    DETECTIONS_PATH: Path = Field(default_factory=lambda: Path("storage/detections"))
    REPORTS_PATH: Path = Field(default_factory=lambda: Path("storage/reports"))
    UPLOADS_PATH: Path = Field(default_factory=lambda: Path("storage/uploads"))
    
    # Model path
    MODEL_PATH: Path = Field(
        default_factory=lambda: Path("C:/Users/samir/AdVisionAI/models/trained/v3/weights/best.pt")
    )
    
    # Upload settings
    MAX_CONTENT_LENGTH: str = "50MB"
    
    # Cloud storage (opcional)
    ORACLE_BUCKET_NAME: Optional[str] = os.getenv("ORACLE_BUCKET_NAME")
    ORACLE_REGION: Optional[str] = os.getenv("ORACLE_REGION")
    ORACLE_NAMESPACE: Optional[str] = os.getenv("ORACLE_NAMESPACE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context):
        # Crear directorios necesarios
        for path in [self.STORAGE_PATH, self.DETECTIONS_PATH,
                    self.REPORTS_PATH, self.UPLOADS_PATH]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Verificar existencia del modelo
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at: {self.MODEL_PATH}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Base directory: {self.BASE_DIR}\n"
                f"Project root: {self.PROJECT_ROOT}"
            )

settings = Settings()