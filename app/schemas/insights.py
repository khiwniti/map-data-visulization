from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class InsightCategory(str, Enum):
    product = "product"
    place = "place"
    price = "price"
    promotion = "promotion"

class InsightRequest(BaseModel):
    project_id: str
    category: InsightCategory

class InsightBase(BaseModel):
    title: str
    description: str
    category: str
    severity: str
    impact: str
    actionable: bool

class InsightCreate(InsightBase):
    project_id: str

class InsightInDB(InsightBase):
    id: str
    project_id: str
    date_generated: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

class InsightResponse(BaseModel):
    data: InsightInDB
    meta: Dict[str, Any] = {}