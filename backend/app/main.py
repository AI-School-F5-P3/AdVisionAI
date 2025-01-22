from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import auth, detection, reports
from app.db.session import init_db

app = FastAPI(
    title="Logo Detection API",
    description="API para detección de logos en imágenes y videos",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(detection.router, prefix="/api/v1/detection", tags=["detection"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    return {"message": "Logo Detection API v1.0"}