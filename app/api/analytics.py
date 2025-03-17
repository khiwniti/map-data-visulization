from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db

router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_analytics(db: Session = Depends(get_db)):
    return {"message": "Dashboard analytics data"}

@router.get("/reports")
async def get_analytical_reports(db: Session = Depends(get_db)):
    return {"message": "Analytical reports data"}