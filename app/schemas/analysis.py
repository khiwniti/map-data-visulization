from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.schemas.projects import CoordinatesModel

class AnalysisRequest(BaseModel):
    project_id: str
    location: str
    cuisine: str
    radius: float = Field(ge=0.5, le=5.0)
    coordinates: CoordinatesModel

class AnalysisBase(BaseModel):
    project_id: str
    status: str
    parameters: Dict[str, Any]
    created_at: datetime
    estimated_completion_time: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

class AnalysisCreate(AnalysisBase):
    pass

class AnalysisInDB(AnalysisBase):
    id: str

    class Config:
        from_attributes = True  # Updated from orm_mode = True

class AnalysisResponse(BaseModel):
    data: AnalysisInDB
    meta: Dict[str, Any] = {}