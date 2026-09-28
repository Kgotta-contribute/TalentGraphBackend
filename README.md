# 🚀 TalentGraph Backend

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![PostgreSQL](https://img.shields.io/badge/Supabase-pgvector-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-BGE--M3-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-m3)

**TalentGraph Backend** is an enterprise-grade, asynchronous AI recruitment intelligence engine. Built with **FastAPI**, **LangGraph**, and **Supabase (PostgreSQL + pgvector)**, it orchestrates a deterministic multi-agent pipeline that analyzes job descriptions, parses resumes, audits real-world GitHub portfolios via MCP, performs high-dimensional vector retrieval, and outputs transparent, auditable candidate ranking dossiers.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [The 6-Agent Pipeline](#-the-6-agent-pipeline)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start (Local Development)](#-quick-start-local-development)
- [Environment Variables](#-environment-variables)
- [Database Migrations & Seeding](#-database-migrations--seeding)
- [API Reference](#-api-reference)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Production Deployment (Railway)](#-production-deployment-railway)

---

## 🏛 Architecture Overview

TalentGraph replaces opaque ATS scoring with a **verifiable, multi-agent evaluation framework**:

```
[ Job Description ] ───► Agent 1 (JD Analyzer) ──┐
                                                 ▼
[ Candidate Resume ] ──► Agent 2 (Profile Parser) ──► Agent 3 (Requirement Verifier)
                                                           │ (pgvector RAG + BGE-M3)
[ GitHub Profile ] ───► Agent 6 (GitHub MCP Engine) ───────┤
                                                           ▼
                                                Agent 4 (Deterministic Scoring)
                                                 [ NO LLM • Pure Mathematical Logic ]
                                                           │
                                                           ▼
                                                Agent 5 (Executive Dossier Agent)
                                                           │
                                                           ▼
                                                [ Structured Interview Report & Rankings ]
```

---

## 🤖 The 6-Agent Pipeline

### 1. Agent 1: Job Description Analyzer (`app/agents/jd_analyzer.py`)
- Ingests raw job descriptions.
- Extracts structured criteria: mandatory skills, preferred qualifications, experience target years, domain tags, responsibilities, and education requirements.
- Standardizes industry terminology (e.g., `K8s` ➔ `Kubernetes`, `Postgres` ➔ `PostgreSQL`).

### 2. Agent 2: Candidate Profile Extractor (`app/agents/resume_parser.py`)
- Parses unstructured resume text into a strict Pydantic model (`CandidateProfile`).
- Extracts timeline chronologies, normalized skill sets, project metadata, education, and links.
- Uses regex pre-extractors for high-precision GitHub and LinkedIn profile discovery.

### 3. Agent 3: Requirement Verifier (`app/agents/requirement_verifier.py`)
- Employs hybrid RAG: queries PostgreSQL `document_chunks` using **BAAI/bge-m3 (1024-dim dense vectors)** cosine similarity.
- Classifies each requirement into `MATCHED`, `PARTIAL`, `MISSING`, or `UNKNOWN`.
- Eliminates hallucinated candidate claims by enforcing explicit source attribution (`RESUME`, `PROJECT`, `GITHUB`, `EDUCATION`).

### 4. Agent 4: Deterministic Scoring & Ranking Engine (`app/ranking/scoring.py`)
- **Zero LLM hallucinations.** Pure mathematical, auditable evaluation.
- Weighted scoring formula:
  $$\text{Final Score} = w_1(\text{Tech}) + w_2(\text{Tenure}) + w_3(\text{JDSim}) + w_4(\text{Projects}) + w_5(\text{Edu})$$
- Default Weights:
  - Technical Skills: **40%**
  - Experience & Tenure: **25%**
  - Semantic JD Similarity: **20%**
  - Project Relevance: **10%**
  - Education & Certifications: **5%**
- Maps scores to confidence tiers: `Strongly Recommended`, `Recommended`, `Interview Candidate`, `Weak Match`, `Not Recommended`.

### 5. Agent 5: Executive Dossier Synthesizer (`app/agents/report_generator.py`)
- Generates comprehensive recruiter dossiers.
- Surfaces key candidate strengths, explicit skill gaps, risk factors, and ramp-up considerations.
- Generates 4–5 targeted technical interview probes with rationales and keywords mapped directly to candidate evidence and identified gaps.

### 6. Agent 6: GitHub MCP Portfolio Intelligence (`app/agents/github_intelligence_agent.py`)
- Connects directly to the GitHub API via Model Context Protocol (MCP).
- Audits public repositories for genuine engineering depth: commit frequency, code topology, dependency health, test coverage, and languages.
- Detects shallow forks vs. original production-grade code.

---

## 🛠 Tech Stack

| Component | Technology | Rationale |
|:---|:---|:---|
| **API Framework** | FastAPI (Python 3.11+) | Asynchronous, auto-generates OpenAPI docs, native Pydantic v2 validation. |
| **Agent Orchestration** | LangGraph & LangChain | Stateful, cyclical workflow execution with SSE progress streaming. |
| **Primary LLM** | Groq (`openai/gpt-oss-120b`) | Sub-second inference latency with fallback support for OpenAI & Anthropic. |
| **Embeddings** | Hugging Face Serverless (`BAAI/bge-m3`) | 8,192 token context window, 1024 dense dimensions, multilingual. |
| **Database** | Supabase (PostgreSQL 15+) | Cloud relational storage with `pgvector` extension for cosine search. |
| **ORM & Migrations**| SQLAlchemy 2.0 (asyncpg) + Alembic | Fully asynchronous database operations and declarative schema versioning. |
| **Deployment** | Docker & Railway | Containerized runtime with automatic port binding and scaling. |

---

## 📂 Project Structure

```
TalentGraphBackend/
├── .env.example                # Template for environment configuration
├── .gitignore                  # Production Git ignore rules
├── alembic.ini                 # Alembic migration configuration
├── Dockerfile                  # Optimized container build
├── Procfile                    # Web process command for cloud PaaS
├── pyproject.toml              # Build backend & packaging metadata
├── railway.json                # Railway deployment configuration
├── requirements.txt            # Locked runtime dependencies
├── README.md                   # Project documentation
│
├── alembic/                    # Database migration scripts
│   ├── env.py
│   └── versions/
│       └── bb40b14047bd_initial_schema.py
│
├── app/                        # Application source code
│   ├── main.py                 # FastAPI initialization and middleware
│   ├── agents/                 # Autonomous agent implementations
│   │   ├── jd_analyzer.py      # Agent 1: Job Description Analyzer
│   │   ├── resume_parser.py    # Agent 2: Resume Profile Extractor
│   │   ├── requirement_verifier.py # Agent 3: Requirement Verifier
│   │   ├── report_generator.py # Agent 5: Dossier Generator
│   │   └── github_intelligence_agent.py # Agent 6: GitHub MCP Agent
│   ├── api/                    # REST API routers
│   │   ├── router.py           # Unified v1 router
│   │   ├── mandates.py         # Mandates & JD endpoints
│   │   ├── candidates.py       # Candidate ingestion & evaluations
│   │   ├── github_mcp.py       # GitHub MCP trigger endpoints
│   │   └── analysis.py         # SSE live streaming events
│   ├── core/                   # Configuration, rate limits & LLM factory
│   │   ├── config.py           # Pydantic BaseSettings & env parsing
│   │   ├── deps.py             # FastAPI dependency injections
│   │   ├── llm.py              # Multi-provider LLM factory (Groq/OpenAI/Anthropic)
│   │   └── rate_limiter.py     # Sliding window rate limiters
│   ├── db/                     # Database sessions & Declarative Base
│   │   ├── base.py
│   │   └── session.py
│   ├── graph/                  # LangGraph state machine workflow
│   │   ├── state.py
│   │   └── workflow.py
│   ├── mcp/                    # Model Context Protocol clients
│   │   └── github_client.py    # GitHub REST/GraphQL async client
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── analysis.py
│   │   ├── candidate.py
│   │   ├── document.py         # DocumentChunk with pgvector Vector(1024)
│   │   ├── evaluation.py
│   │   └── mandate.py
│   ├── ranking/                # Agent 4: Deterministic Scoring
│   │   └── scoring.py
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── candidate.py
│   │   ├── evaluation.py
│   │   ├── github.py
│   │   ├── job_requirements.py
│   │   └── report.py
│   └── services/               # Vector search & embedding client
│       ├── embedding.py        # Hugging Face BGE-M3 client
│       └── vector_search.py    # pgvector cosine similarity search
│
├── scripts/                    # Utilities & sample seed data
│   ├── seed_5_templates.py
│   └── seed_demo_data.py
│
└── tests/                      # Pytest automated test suite
    ├── test_schemas.py
    └── test_scoring.py
```

---

## ⚡ Quick Start (Local Development)

### 1. Prerequisites
- Python 3.11 or higher
- PostgreSQL instance with `pgvector` enabled (e.g. Supabase)
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/Kgotta-contribute/TalentGraphBackend.git
cd TalentGraphBackend

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure `.env`
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### 4. Run Database Migrations
```bash
alembic upgrade head
```

### 5. Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Open [http://localhost:8000](http://localhost:8000) (Returns API Status)
- Open [http://localhost:8000/docs](http://localhost:8000/docs) (Interactive Swagger UI)
- Open [http://localhost:8000/health](http://localhost:8000/health) (Health Check)

---

## 🔑 Environment Variables

| Variable | Description | Example / Default |
|:---|:---|:---|
| `APP_NAME` | Name of the service | `TalentAgent` |
| `AUTH_MODE` | Authentication mode (`dev` or `puter`) | `dev` |
| `CORS_ORIGINS` | Allowed origins (JSON array or comma-separated) | `["http://localhost:5173", "*"]` |
| `DATABASE_URL` | Async PostgreSQL connection string | `postgresql+asyncpg://user:pass@host:5432/db` |
| `SUPABASE_URL` | Supabase project URL | `https://xxxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role secret | `sb_secret_xxxx` |
| `EMBEDDING_MODEL` | Hugging Face embedding model | `BAAI/bge-m3` |
| `EMBEDDING_DIMENSIONS`| Dimension count of dense embeddings | `1024` |
| `HF_TOKEN` | Hugging Face API Token | `hf_xxxx` |
| `LLM_PROVIDER` | Active LLM (`groq`, `openai`, or `anthropic`) | `groq` |
| `GROQ_API_KEY` | Groq Cloud API key | `gsk_xxxx` |
| `GROQ_MODEL` | Groq Model ID | `openai/gpt-oss-120b` |
| `GROQ_BASE_URL` | Groq OpenAI-compatible base URL | `https://api.groq.com/openai/v1` |
| `GITHUB_TOKEN` | GitHub Personal Access Token for MCP | `github_pat_xxxx` |

---

## 🗄 Database Migrations & Seeding

### Apply Migrations
```bash
alembic upgrade head
```

### Create New Migration
```bash
alembic revision --autogenerate -m "describe_changes"
```

### Seed Demo Mandates & Candidates
```bash
python scripts/seed_5_templates.py
python scripts/seed_demo_data.py
```

---

## 📡 API Reference

### System
- `GET /` - Root status and API service verification.
- `GET /health` - Health check endpoint.
- `GET /docs` - Interactive OpenAPI documentation (Swagger).

### Mandates (`/api/v1/mandates`)
- `POST /` - Create a new recruitment mandate.
- `GET /` - List all mandates for the current recruiter.
- `GET /{mandate_id}` - Retrieve details and extracted requirements for a mandate.
- `PUT /{mandate_id}/job-description` - Update raw job description text.
- `POST /{mandate_id}/analyze-jd` - Trigger Agent 1 to extract structured requirements.
- `GET /{mandate_id}/candidates` - Retrieve all candidates associated with a mandate.
- `POST /{mandate_id}/verify` - Trigger Agent 3 verification across all candidates.
- `POST /{mandate_id}/rank` - Run Agent 4 deterministic scoring.
- `GET /{mandate_id}/ranking` - Retrieve ranked leaderboard for a mandate.
- `PUT /{mandate_id}/scoring-profile` - Dynamically adjust weight distribution.
- `POST /{mandate_id}/analysis-runs` - Trigger full LangGraph autonomous pipeline.

### Candidates (`/api/v1/candidates`)
- `POST /import-from-resume` - Ingest candidate from ResumeIQ / Puter storage.
- `GET /{candidate_id}` - Retrieve parsed candidate profile.
- `POST /{candidate_id}/github-analysis` - Trigger Agent 6 GitHub MCP audit.
- `GET /{candidate_id}/github-analysis` - Retrieve GitHub portfolio evidence.
- `POST /{candidate_id}/report` - Generate Agent 5 recruitment dossier.
- `GET /{candidate_id}/report` - Retrieve existing recruitment dossier.

### Real-Time Streaming (`/api/v1/analysis-runs`)
- `GET /{run_id}/events` - Server-Sent Events (SSE) stream reporting real-time progress across all 6 agents.

---

## 🧪 Testing & Quality Assurance

Run the comprehensive test suite with `pytest`:

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run deterministic scoring tests
pytest tests/test_scoring.py -v

# Run schema validation tests
pytest tests/test_schemas.py -v
```

---

## 🚢 Production Deployment (Railway)

The repository includes native support for [Railway](https://railway.app) via `Dockerfile` and `railway.json`.

1. **Push your code to GitHub**:
   ```bash
   git push origin main
   ```
2. **Create a Railway Project**:
   - Navigate to [railway.app](https://railway.app).
   - Select **New Project** ➔ **Deploy from GitHub repo**.
   - Choose `TalentGraphBackend`.
3. **Configure Environment Variables**:
   - Go to your service ➔ **Variables** tab.
   - Click **RAW Editor** and paste all required keys from `.env.example`.
4. **Generate Public Domain**:
   - Go to **Settings** ➔ **Networking** ➔ Click **Generate Domain**.
5. **Verify**:
   - Access `https://<your-railway-domain>/health`.
