import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.db.base import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mandate_id = Column(UUID(as_uuid=True), nullable=True)
    candidate_id = Column(UUID(as_uuid=True), nullable=True)
    source_type = Column(String, nullable=False)  # JOB_DESCRIPTION | RESUME | EXPERIENCE | PROJECT | GITHUB_README | GITHUB_CODE
    source_id = Column(String, nullable=True)  # ID of the parent object
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)
    embedding = Column(Vector(1024), nullable=True)  # BAAI/bge-m3 = 1024 dims
    embedding_model = Column(String, default="BAAI/bge-m3")
    created_at = Column(DateTime, default=datetime.utcnow)
