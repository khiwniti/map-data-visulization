from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base
from datetime import datetime

class Insight(Base):
    __tablename__ = "insights"

    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id"))
    title = Column(String)
    description = Column(String)
    category = Column(String)
    severity = Column(String)
    impact = Column(String)
    actionable = Column(Boolean, default=True)
    date_generated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationship with Project
    project = relationship("Project", back_populates="insights")