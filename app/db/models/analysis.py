from sqlalchemy import Column, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base
from datetime import datetime

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True)
    project_id = Column(String, ForeignKey("projects.id"))
    status = Column(String, nullable=False)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    estimated_completion_time = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)
    results = Column(JSON, nullable=True)

    # Relationship with Project
    project = relationship("Project", back_populates="analyses")