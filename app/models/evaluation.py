import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class CandidateEvaluation(Base):
    __tablename__ = "candidate_evaluations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mandate_id = Column(UUID(as_uuid=True), ForeignKey("mandates.id"), nullable=False)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False)
    # Verification (Agent 3)
    verification_result = Column(JSON, nullable=True)
    tech_coverage_pct = Column(Float, nullable=True)
    # GitHub (conditional)
    github_analysis = Column(JSON, nullable=True)
    github_verified = Column(Boolean, default=False)
    # Score (Agent 4 - DETERMINISTIC)
    score_technical = Column(Float, nullable=True)
    score_experience = Column(Float, nullable=True)
    score_jd_similarity = Column(Float, nullable=True)
    score_projects = Column(Float, nullable=True)
    score_education = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    tier = Column(String, nullable=True)  # Strongly Recommended | Recommended | etc.
    scoring_weights = Column(JSON, nullable=True)  # weights used for this evaluation
    # Report (Agent 5)
    report = Column(JSON, nullable=True)
    # Metadata
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
