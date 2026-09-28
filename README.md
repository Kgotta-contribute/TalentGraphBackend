# TalentGraph Backend

FastAPI and LangGraph backend for the TalentGraph AI candidate verification and recruitment intelligence platform.

## Features
- **FastAPI Async Engine**: RESTful endpoints and Server-Sent Events (SSE) streaming.
- **5-Agent Autonomous Pipeline**: JD Analyzer, Resume Parser, Requirement Verifier, Deterministic Ranker, and Recruitment Dossier Synthesizer.
- **Agent 6 GitHub MCP Harness**: Multi-agent repository analysis (topology, dependencies, RAG, security, CI/CD, code quality, and chat).
- **PostgreSQL & pgvector**: Vector similarity matching using BAAI/bge-m3 embeddings.
- **Automated Regression Suite**: Invariant validation and schema integrity assertions via pytest.

---

## Local Development

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Configure environment:**
   Ensure `.env` exists with your keys (see `.env.example`).

4. **Run the automated test suite:**
   ```bash
   pytest tests/ -v
   ```

5. **Start development server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   Verify at [http://localhost:8000/health](http://localhost:8000/health).

---

## Deploying to Railway

1. **Push this folder to a GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for TalentGraphBackend"
   git branch -M main
   git remote add origin https://github.com/<your-username>/talent-graph-backend.git
   git push -u origin main
   ```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app) and create a **New Project**.
   - Select **Deploy from GitHub repo** and choose `talent-graph-backend`.
   - Railway will automatically detect the `Dockerfile` and `railway.json`.
   - In **Variables**, add all keys from your `.env`:
     - `DATABASE_URL`
     - `SUPABASE_URL`
     - `SUPABASE_SERVICE_ROLE_KEY`
     - `GROQ_API_KEY`
     - `GROQ_MODEL`
     - `GROQ_BASE_URL`
     - `EMBEDDING_MODEL`
     - `GITHUB_TOKEN`
     - `CORS_ORIGINS`: Set to your deployed Vercel frontend URL (e.g., `https://your-frontend.vercel.app`).
   - In **Settings** > **Networking**, click **Generate Domain** to get your public backend URL.
