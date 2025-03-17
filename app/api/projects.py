from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.db.models.projects import Project, ProjectTeamMember, ProjectMilestone
from app.schemas.projects import (
    ProjectCreate, ProjectResponse, ProjectUpdate,
    MilestoneCreate, MilestoneResponse, CompetitorCreate, CompetitorResponse
)
from app.core.security import get_current_user
from app.db.models.users import User

router = APIRouter()

@router.get("/", response_model=List[ProjectResponse])
async def get_projects(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Only return projects owned by the current user
    projects = db.query(Project).filter(Project.user_id == current_user.id).offset(skip).limit(limit).all()
    return [{"data": project, "meta": {}} for project in projects]

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project: ProjectCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Create project with current user as owner
    db_project = Project(
        **project.dict(exclude={"team_members", "coordinates"}),
        user_id=current_user.id,
        lat=project.coordinates.lat if project.coordinates else None,
        lng=project.coordinates.lng if project.coordinates else None
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    # Add team members if provided
    if project.team_members:
        for user_id in project.team_members:
            team_member = ProjectTeamMember(project_id=db_project.id, user_id=user_id)
            db.add(team_member)
        db.commit()
    
    return {"data": db_project, "meta": {}}

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Verify project ownership
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    return {"data": project, "meta": {}}

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_update: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Verify project ownership
    if db_project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this project")
    
    # Update project fields
    update_data = project_update.dict(exclude_unset=True)
    if "coordinates" in update_data and update_data["coordinates"]:
        update_data["lat"] = update_data["coordinates"].lat
        update_data["lng"] = update_data["coordinates"].lng
        del update_data["coordinates"]
    
    for key, value in update_data.items():
        setattr(db_project, key, value)
    
    db.commit()
    db.refresh(db_project)
    return {"data": db_project, "meta": {}}

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Verify project ownership
    if db_project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this project")
    
    db.delete(db_project)
    db.commit()
    return None

@router.post("/{project_id}/milestones", response_model=MilestoneResponse)
async def create_milestone(
    project_id: int,
    milestone: MilestoneCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check if project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check authorization
    if current_user.role not in ["admin", "staff"]:
        raise HTTPException(status_code=403, detail="Not authorized to create milestones")
    
    # Create milestone
    db_milestone = ProjectMilestone(
        **milestone.dict(),
        project_id=project_id
    )
    db.add(db_milestone)
    db.commit()
    db.refresh(db_milestone)
    
    return {"data": db_milestone, "meta": {}}

@router.get("/{project_id}/milestones", response_model=List[MilestoneResponse])
async def get_project_milestones(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check if project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    milestones = db.query(ProjectMilestone).filter(ProjectMilestone.project_id == project_id).all()
    return [{"data": milestone, "meta": {}} for milestone in milestones]
