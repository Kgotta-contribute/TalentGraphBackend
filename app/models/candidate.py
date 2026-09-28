import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Source: imported from ResumeIQ or uploaded directly
    resumeiq_resume_id = Column(String, nullable=True)  # Puter KV key
    puter_file_path = Column(String, nullable=True)     # Puter storage path
    raw_text = Column(Text, nullable=True)
    resumeiq_audit_json = Column(JSON, nullable=True)   # Existing ResumeIQ diagnostic
    # Agent 2 output
    profile = Column(JSON, nullable=True)  # CandidateProfile structured data
    full_name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    github_url = Column(String, nullable=True)
    linkedin_url = Column(String, nullable=True)
    current_title = Column(String, nullable=True)
    years_experience = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
