from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.db.models.projects import Project
# Import Analysis from analysis.py instead of projects.py
from app.db.models.analysis import Analysis
from app.schemas.analysis import AnalysisCreate, AnalysisResponse, AnalysisRequest
from app.core.security import get_current_user
from app.db.models.users import User
import uuid
from datetime import datetime, timedelta

router = APIRouter()

# Mock background task to simulate analysis processing
async def process_analysis(analysis_id: str, db: Session):
    # In a real implementation, this would perform actual analysis
    # For now, we'll just update the status after a delay
    import asyncio
    await asyncio.sleep(5)  # Simulate processing time
    
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if analysis:
        analysis.status = "completed"
        analysis.completed_at = datetime.utcnow()
        db.commit()

@router.post("/", response_model=AnalysisResponse)
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    # Create analysis record
    analysis_id = str(uuid.uuid4())
    analysis = Analysis(
        id=analysis_id,
        project_id=request.project_id,
        status="processing",
        parameters={
            "location": request.location,
            "cuisine": request.cuisine,
            "radius": request.radius,
            "coordinates": {
                "lat": request.coordinates.lat,
                "lng": request.coordinates.lng
            }
        },
        created_at=datetime.utcnow(),
        estimated_completion_time=datetime.utcnow() + timedelta(minutes=5)
    )
    
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Start background processing
    background_tasks.add_task(process_analysis, analysis_id, db)
    
    return {
        "data": analysis,
        "meta": {
            "estimatedCompletionTime": "5 minutes"
        }
    }

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Verify project ownership
    project = db.query(Project).filter(Project.id == analysis.project_id).first()
    if not project or project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this analysis")
    
    return {"data": analysis, "meta": {}}