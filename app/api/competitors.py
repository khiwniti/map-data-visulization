from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.db.models.projects import Project, Competitor
from app.schemas.projects import CompetitorCreate, CompetitorResponse, CompetitorUpdate
from app.core.security import get_current_user
from app.db.models.users import User

router = APIRouter()

@router.get("/{project_id}/competitors", response_model=List[CompetitorResponse])
async def get_competitors(
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
    
    competitors = db.query(Competitor).filter(Competitor.project_id == project_id).all()
    return [{"data": competitor, "meta": {}} for competitor in competitors]

@router.post("/{project_id}/competitors", response_model=CompetitorResponse)
async def create_competitor(
    project_id: str,
    competitor: CompetitorCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to add competitors to this project")
    
    # Create competitor
    db_competitor = Competitor(
        **competitor.dict(exclude={"coordinates"}),
        project_id=project_id,
        lat=competitor.coordinates.lat if competitor.coordinates else None,
        lng=competitor.coordinates.lng if competitor.coordinates else None
    )
    db.add(db_competitor)
    db.commit()
    db.refresh(db_competitor)
    
    return {"data": db_competitor, "meta": {}}

@router.get("/{project_id}/competitors/{competitor_id}", response_model=CompetitorResponse)
async def get_competitor(
    project_id: str,
    competitor_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Verify project exists and user has access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    competitor = db.query(Competitor).filter(
        Competitor.id == competitor_id,
        Competitor.project_id == project_id
    ).first()
    
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")
    
    return {"data": competitor, "meta": {}}