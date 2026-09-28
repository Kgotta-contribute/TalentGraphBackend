import asyncio
import uuid
from app.db.session import AsyncSessionLocal
from app.models.mandate import Mandate
from sqlalchemy import select

MANDATE_ID = uuid.UUID('a0000000-0000-0000-0000-000000000001')

RAW_JD = """TEKION COMPANY

We are seeking a Senior Full-Stack & AI Systems Engineer to architect and build our next-generation multi-agent LLM platform.

Key Responsibilities:
- Architect and maintain asynchronous Python/FastAPI microservices.
- Build multi-agent LLM & RAG retrieval systems with LangGraph/LangChain.
- Develop responsive React & TypeScript interfaces.
- Optimize vector database indexes and semantic search retrieval (FAISS, pgvector).
- Deploy containerized services to cloud infrastructure with CI/CD.

Mandatory Technical Skills:
Python, FastAPI, LangGraph, LangChain, React, TypeScript, Vector Search, FAISS, Docker, PostgreSQL.

Preferred Qualifications:
AWS, Kubernetes, Redis, Celery, Sentence-Transformers, ChromaDB, CI/CD, Tailwind CSS.

Experience Target:
4+ Years (Senior). Bachelor's or Master's in Computer Science or related quantitative field."""

async def update():
    async with AsyncSessionLocal() as s:
        res = await s.execute(select(Mandate).where(Mandate.id == MANDATE_ID))
        m = res.scalar_one_or_none()
        if m:
            m.title = "Senior Full-Stack & AI Systems Engineer"
            m.company = "Tekion"
            m.raw_jd = RAW_JD
            if m.job_requirements:
                m.job_requirements["role"] = "Senior Full-Stack & AI Systems Engineer"
            await s.commit()
            print("Successfully updated a0000000-...0001 to Tekion in the database!")

if __name__ == "__main__":
    asyncio.run(update())
