from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.schemas import schemas
from app.auth.auth import get_current_user
from app.db.models import Detection, Report

router = APIRouter()

@router.get("/reports/{detection_id}", response_model=schemas.Report)
async def get_report(
    detection_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    # Verificar acceso
    detection = db.query(Detection).filter(
        Detection.id == detection_id,
        Detection.user_id == current_user.id
    ).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    report = db.query(Report).filter(Report.detection_id == detection_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return report

@router.get("/reports/user/list", response_model=List[schemas.Report])
async def list_user_reports(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    reports = db.query(Report).join(Detection).filter(
        Detection.user_id == current_user.id
    ).offset(skip).limit(limit).all()
    
    return reports