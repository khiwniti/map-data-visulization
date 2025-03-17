from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Text, 
    Date, 
    DateTime, 
    Float, 
    Numeric, 
    Boolean, 
    ForeignKey
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    start_date = Column(Date)
    end_date = Column(Date)
    budget = Column(Numeric(12, 2))
    status = Column(String(20))
    priority = Column(String(20))
    manager_id = Column(Integer, ForeignKey("users.id"))
    completion_percentage = Column(Numeric(5, 2), default=0)
    days_remaining = Column(Integer)
    budget_utilization = Column(Numeric(5, 2), default=0)
    burn_rate = Column(Numeric(10, 2))
    risk_score = Column(Numeric(5, 2))
    on_track_status = Column(Boolean, default=True)
    efficiency_score = Column(Numeric(5, 2))
    
    # Relationships
    insights = relationship("Insight", back_populates="project", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="project", cascade="all, delete-orphan")
    competitors = relationship("Competitor", back_populates="project", cascade="all, delete-orphan")

class ProjectTeamMember(Base):
    __tablename__ = "project_team_members"
    
    project_id = Column(Integer, ForeignKey("projects.id"), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)

class ProjectMilestone(Base):
    __tablename__ = "project_milestones"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    title = Column(String(100), nullable=False)
    description = Column(Text)
    due_date = Column(Date)
    completed = Column(Boolean, default=False)
class Competitor(Base):
    __tablename__ = "competitors"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    location = Column(String)
    rating = Column(Float)
    price_category = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to project
    project = relationship("Project", back_populates="competitors")
