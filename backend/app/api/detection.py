from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
from app.db.session import get_db
from app.schemas import schemas
from app.services.detector import DetectorService
from app.services.reporting import ReportingService
from app.auth.auth import get_current_user
from app.db.models import Detection, DetectionResult, Report
from pathlib import Path
import aiofiles
import cv2
import numpy as np

router = APIRouter()
detector = DetectorService()
reporter = ReportingService()

@router.post("/detect/image", response_model=schemas.Detection)
async def detect_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    # Crear directorios de almacenamiento
    detection_dir = Path(f"storage/detections/{current_user.id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    detection_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Leer y procesar imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detectar
        detections = await detector.process_image(image, detection_dir)

        # Guardar en DB
        db_detection = Detection(
            user_id=current_user.id,
            video_name=file.filename,
            source_type="image",
            storage_path=str(detection_dir)
        )
        db.add(db_detection)
        db.commit()

        # Guardar resultados
        for det in detections:
            db_result = DetectionResult(
                detection_id=db_detection.id,
                brand=det["class"],
                confidence=det["confidence"],
                bbox_coordinates=str(det["bbox"]),
                bbox_image_path=det["crop_path"]
            )
            db.add(db_result)
        db.commit()

        return db_detection

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/detect/video", response_model=schemas.Detection)
async def detect_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    try:
        print("\n=== INICIANDO DETECCIÓN DE VIDEO ===")
        detection_dir = Path(f"storage/detections/{current_user.id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        detection_dir.mkdir(parents=True, exist_ok=True)
        video_path = detection_dir / file.filename
        
        print(f"Guardando video en: {video_path}")
        async with aiofiles.open(video_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        print("Procesando video...")
        results = await detector.process_video(video_path, detection_dir)
        print(f"Resultados: {str(results)}")  # Convertir a string para debug

        # Crear el objeto Detection
        db_detection = Detection(
            user_id=current_user.id,
            video_name=file.filename,
            source_type="video",
            storage_path=str(detection_dir),
            duration_seconds=float(results["duration"]),
            total_frames=int(results["total_frames"]),
            detection_time=datetime.utcnow()  # Añadir este campo
        )
        db.add(db_detection)
        db.commit()
        db.refresh(db_detection)  # Esto asegura que tengamos el ID
       
        # Procesar resultados de detección
        if "brand_stats" in results:
            for brand, stats in results["brand_stats"].items():
                db_result = DetectionResult(
                    detection_id=db_detection.id,
                    brand=str(brand),
                    confidence=float(stats["average_confidence"]),
                    screen_time_percentage=float(stats["screen_time_percentage"]),
                    bbox_coordinates="",  # Añadir este campo
                    bbox_image_path=""    # Añadir este campo
                )
                db.add(db_result)
            db.commit()

        return db_detection

    except Exception as e:
        print(f"Error en detect_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
  
@router.post("/detect/youtube", response_model=schemas.Detection)
async def detect_youtube(
    url: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    try:
        detection_dir = Path(f"storage/detections/{current_user.id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        detection_dir.mkdir(parents=True, exist_ok=True)

        # Procesar video de YouTube
        results = await detector.process_youtube(url, detection_dir)

        # Crear el registro de detección con todos los campos requeridos
        db_detection = Detection(
            user_id=current_user.id,
            video_name=f"youtube_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            video_url=url,
            source_type="youtube",
            storage_path=str(detection_dir),
            duration_seconds=float(results["duration"]),
            total_frames=int(results["total_frames"]),
            detection_time=datetime.utcnow()  # Aseguramos que este campo esté presente
        )
        db.add(db_detection)
        db.flush()  # Para obtener el ID antes del commit
        db.refresh(db_detection)

        # Procesar y guardar los resultados de detección
        if "brand_stats" in results:
            for brand, stats in results["brand_stats"].items():
                detection_result = DetectionResult(
                    detection_id=db_detection.id,
                    brand=str(brand),
                    confidence=float(stats["average_confidence"]),
                    screen_time_percentage=float(stats["screen_time_percentage"]),
                    bbox_coordinates="",  # Campo requerido
                    frame_number=0        # Si es requerido
                )
                db.add(detection_result)
            db.commit()

        # Crear y guardar el reporte
        if "youtube_info" in results:
            report = Report(
                detection_id=db_detection.id,
                content=f"YouTube video analysis - {results['youtube_info'].get('title', '')}",
                summary_stats=str(results["brand_stats"]),
                created_at=datetime.utcnow()
            )
            db.add(report)
            db.commit()

        return db_detection

    except Exception as e:
        print(f"Error en detect_youtube: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
