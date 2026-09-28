import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mandate_id = Column(UUID(as_uuid=True), ForeignKey("mandates.id"))
    run_type = Column(String, default="full")  # full | jd_only | rank_only
    status = Column(String, default="pending")  # pending | running | completed | failed
    # LangGraph state
    agent1_status = Column(String, default="pending")
    agent2_status = Column(String, default="pending")
    agent3_status = Column(String, default="pending")
    github_status = Column(String, default="not_applicable")
    agent4_status = Column(String, default="pending")
    agent5_status = Column(String, default="pending")
    # Metadata
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    errors = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
