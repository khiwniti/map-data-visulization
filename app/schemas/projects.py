from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class CoordinatesModel(BaseModel):
    lat: float
    lng: float

class ProjectBase(BaseModel):
    name: str
    cuisine: str
    location: str
    address: Optional[str] = None
    radius: float = Field(ge=0.5, le=5.0)
    coordinates: Optional[CoordinatesModel] = None

class ProjectCreate(ProjectBase):
    team_members: Optional[List[str]] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    cuisine: Optional[str] = None
    location: Optional[str] = None
    address: Optional[str] = None
    radius: Optional[float] = Field(default=None, ge=0.5, le=5.0)
    coordinates: Optional[CoordinatesModel] = None

class ProjectInDB(ProjectBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    lat: Optional[float] = None
    lng: Optional[float] = None

    class Config:
        orm_mode = True

class ProjectResponse(BaseModel):
    data: ProjectInDB
    meta: Dict[str, Any] = {}

class CompetitorBase(BaseModel):
    name: str
    cuisine: str
    location: str
    address: Optional[str] = None
    coordinates: Optional[CoordinatesModel] = None
    rating: Optional[float] = None
    price_level: Optional[int] = None
    website: Optional[str] = None

class CompetitorCreate(CompetitorBase):
    pass

class CompetitorUpdate(BaseModel):
    name: Optional[str] = None
    cuisine: Optional[str] = None
    location: Optional[str] = None
    address: Optional[str] = None
    coordinates: Optional[CoordinatesModel] = None
    rating: Optional[float] = None
    price_level: Optional[int] = None
    website: Optional[str] = None

class CompetitorInDB(CompetitorBase):
    id: str
    project_id: str
    created_at: datetime
    updated_at: datetime
    lat: Optional[float] = None
    lng: Optional[float] = None

    class Config:
        orm_mode = True

class CompetitorResponse(BaseModel):
    data: CompetitorInDB
    meta: Dict[str, Any] = {}

class MilestoneBase(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    status: str = "pending"

class MilestoneCreate(MilestoneBase):
    pass

class MilestoneInDB(MilestoneBase):
    id: str
    project_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class MilestoneResponse(BaseModel):
    data: MilestoneInDB
    meta: Dict[str, Any] = {}

