from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.db.models.projects import Project
# Import Insight from insights.py instead of projects.py
from app.db.models.insights import Insight
from app.schemas.insights import InsightCreate, InsightResponse, InsightRequest
from app.core.security import get_current_user
from app.db.models.users import User

router = APIRouter()

@router.post("/generate", response_model=InsightResponse)
async def generate_insights(
    request: InsightRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    # In a real implementation, this would trigger an async job to generate insights
    # For now, we'll create a placeholder insight
    insight = Insight(
        title=f"New {request.category} Insight",
        description=f"This is a placeholder insight for {request.category}",
        category=request.category,
        severity="medium",
        impact="medium",
        actionable=True,
        project_id=request.project_id
    )
    
    db.add(insight)
    db.commit()
    db.refresh(insight)
    
    return {"data": insight, "meta": {"status": "completed"}}

@router.get("/projects/{project_id}/insights", response_model=List[InsightResponse])
async def get_insights(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    insights = db.query(Insight).filter(Insight.project_id == project_id).all()
    return [{"data": insight, "meta": {}} for insight in insights]

@router.get("/projects/{project_id}/insights/{insight_id}", response_model=InsightResponse)
async def get_insight(
    project_id: str,
    insight_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    insight = db.query(Insight).filter(
        Insight.id == insight_id,
        Insight.project_id == project_id
    ).first()
    
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    
    return {"data": insight, "meta": {}}