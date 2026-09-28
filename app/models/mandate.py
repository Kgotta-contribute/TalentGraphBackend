import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class Mandate(Base):
    __tablename__ = "mandates"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recruiter_id = Column(String, nullable=False, index=True)  # Puter user ID
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    raw_jd = Column(Text, nullable=True)
    status = Column(String, default="draft")  # draft, analyzing, active, closed
    job_requirements = Column(JSON, nullable=True)  # Agent 1 output
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
