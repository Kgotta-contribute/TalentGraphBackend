# -*- coding: utf-8 -*-
"""
GitHub Intelligence Harness Agent (Agent 6)

Architecture:
  GitHub URL (supports root repos or subfolders e.g. /tree/main/subfolder) ->
  Agent Planner -> decides tool strategy ->
  [Recursive Git Tree, Code Search via MCP] -> Context Manager ->
  Analysis Agents [Architecture, RAG, Security, CI/CD, etc.] -> Evidence Store ->
  Report Generator
"""
import asyncio
import json
import logging
import re
import time
from typing import TypedDict, Any
from openai import AsyncOpenAI
from app.core.config import settings
from app.mcp.github_client import GitHubMCPClient
from app.core.rate_limiter import groq_rate_limiter

logger = logging.getLogger("talent_agent.github_intelligence")

# ─────────────────────────────────────────────────────────────────────────────
# State Model
# ─────────────────────────────────────────────────────────────────────────────

class GitHubAnalysisState(TypedDict, total=False):
    # Input
    repo_url: str
    owner: str
    repo: str
    branch: str | None
    subpath: str | None
    # Raw MCP data
    repo_metadata: dict
    file_tree: list[dict]
    languages: list[dict]
    readme: str
    manifest_files: dict[str, str]       # path → content
    cicd_files: dict[str, str]           # path → content
    source_code_samples: dict[str, str]  # path → content
    recent_commits: list[dict]
    contributors: list[dict]
    open_issues_count: int
    open_prs_count: int
    # Analysis results
    architecture: dict
    tech_stack: dict
    dependencies: dict
    rag_analysis: dict
    agent_detection: dict
    security: dict
    cicd: dict
    code_quality: dict
    git_activity: dict
    # Observability
    tools_used: int
    files_analyzed: int
    execution_steps: list[dict]          # [{label, status, detail}]
    execution_time_seconds: float
    errors: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────

_llm = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
_sem = asyncio.Semaphore(3)


FALLBACK_MODELS = [
    settings.groq_model,     # Primary configured model (e.g. openai/gpt-oss-120b)
    "qwen/qwen3.8-27b",       # High token throughput and excellent JSON adherence
    "openai/gpt-oss-20b",     # Fast fallback
]


async def _llm_json(system: str, user: str, retries: int = 1) -> dict:
    """Call Groq with JSON enforcement, rate limiting (20 RPM), semaphore gating, model failover, and timeout."""
    async with _sem:
        for model_name in FALLBACK_MODELS:
            for attempt in range(retries + 1):
                try:
                    await groq_rate_limiter.acquire()
                    resp = await asyncio.wait_for(
                        _llm.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.1,
                        ),
                        timeout=25.0,
                    )
                    content = resp.choices[0].message.content.strip()
                    if content:
                        return json.loads(content)
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if "rate limit" in err_msg or "429" in err_msg or "json_validate_failed" in err_msg:
                        # Fail over to the next model immediately if rate limited or invalid
                        break
                    elif attempt == retries:
                        break
                    else:
                        await asyncio.sleep(1.0 * (attempt + 1))
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step(state: GitHubAnalysisState, label: str, status: str = "done", detail: str = "") -> None:
    state.setdefault("execution_steps", []).append(
        {"label": label, "status": status, "detail": detail}
    )


def _inc_tools(state: GitHubAnalysisState, n: int = 1) -> None:
    state["tools_used"] = state.get("tools_used", 0) + n


def _inc_files(state: GitHubAnalysisState, n: int = 1) -> None:
    state["files_analyzed"] = state.get("files_analyzed", 0) + n


# ─────────────────────────────────────────────────────────────────────────────
# URL Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_github_url(url: str) -> tuple[str, str, str | None, str | None]:
    """
    Extracts owner, repo, optional branch, and optional subpath from GitHub URLs.
    Handles:
      https://github.com/owner/repo
      https://github.com/owner/repo/tree/branch/subpath/to/folder
      owner/repo
    """
    clean = url.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]

    pattern = r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    match = re.search(pattern, clean)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        branch = match.group(3)
        subpath = match.group(4)
        return owner, repo, branch, subpath

    parts = clean.split("/")
    if len(parts) == 2 and not clean.startswith("http"):
        return parts[0], parts[1], None, None

    raise ValueError(f"Invalid GitHub repository URL: {url}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 -- Data Collection via MCP
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_data(state: GitHubAnalysisState, client: GitHubMCPClient) -> None:
    owner, repo = state["owner"], state["repo"]
    subpath = state.get("subpath") or ""

    # 1a. Repo metadata
    meta = {}
    try:
        meta = await client.get_repo_metadata(owner, repo)
        state["repo_metadata"] = meta
        _inc_tools(state)
        _step(state, "Repository metadata fetched")
    except Exception as e:
        state.setdefault("errors", []).append(f"metadata: {e}")
        state["repo_metadata"] = {}

    # 1b. Languages
    try:
        langs_raw = await client.get_repo_languages(owner, repo)
        total = sum(langs_raw.values()) or 1
        state["languages"] = [
            {"name": k, "bytes": v, "percentage": round(v / total * 100, 1)}
            for k, v in sorted(langs_raw.items(), key=lambda x: x[1], reverse=True)
        ]
        _inc_tools(state)
        _step(state, "Language distribution analyzed", detail=f"{len(state['languages'])} languages")
    except Exception as e:
        state.setdefault("errors", []).append(f"languages: {e}")
        state["languages"] = []

    # 1c. File tree -- Recursive Git Tree via Git Trees API
    file_tree: list[dict] = []
    try:
        branch_to_use = state.get("branch") or meta.get("default_branch") or "HEAD"
        raw_tree = await client.get_tree_recursive(owner, repo, branch=branch_to_use)
        _inc_tools(state)

        if raw_tree:
            for item in raw_tree:
                p = item.get("path", "")
                is_dir = item.get("type") == "tree"
                file_tree.append({
                    "name": p.split("/")[-1],
                    "type": "dir" if is_dir else "file",
                    "path": p,
                    "size": item.get("size", 0),
                })
        else:
            # Fallback to contents API
            root_items = await client.get_repo_contents(owner, repo, path=subpath)
            _inc_tools(state)
            if isinstance(root_items, list):
                for item in root_items:
                    file_tree.append({
                        "name": item.get("name"),
                        "type": item.get("type"),
                        "path": item.get("path"),
                        "size": item.get("size", 0),
                    })

        state["file_tree"] = file_tree
        _step(state, "Recursive file structure mapped", detail=f"{len(file_tree)} total entries")
    except Exception as e:
        state.setdefault("errors", []).append(f"file_tree: {e}")
        state["file_tree"] = []

    # 1d. README (subpath first, then root)
    try:
        readme = await client.get_readme(owner, repo, subpath=subpath)
        state["readme"] = readme[:8000]
        _inc_tools(state)
        _step(state, "README retrieved", detail=f"{len(readme)} chars")
    except Exception as e:
        logger.warning(f"Error fetching README for {owner}/{repo}: {e}", exc_info=True)
        state["readme"] = ""

    # 1e. Manifest files (searched across entire recursive tree, prioritizing subpath)
    manifest_names = {
        "package.json", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
        "go.mod", "Cargo.toml", "Dockerfile", "docker-compose.yml", "pom.xml",
        "setup.py", "setup.cfg", "poetry.lock", ".python-version",
    }
    manifests: dict[str, str] = {}

    def manifest_priority(item: dict) -> tuple[int, int]:
        p = item.get("path", "")
        if subpath and (p.startswith(f"{subpath}/") or p == subpath):
            return (0, len(p))
        if "/" not in p:
            return (1, len(p))
        return (2, len(p))

    sorted_for_manifests = sorted(file_tree, key=manifest_priority)
    for f in sorted_for_manifests:
        if f.get("type") == "file" and f.get("name") in manifest_names:
            p = f.get("path", "")
            if p not in manifests and len(manifests) < 8:
                try:
                    content = await client.get_file_content(owner, repo, p)
                    if content:
                        manifests[p] = content[:3000]
                        _inc_tools(state)
                        _inc_files(state)
                except Exception as e:
                    logger.warning(f"Error reading manifest {p} for {owner}/{repo}: {e}", exc_info=True)

    state["manifest_files"] = manifests
    if manifests:
        _step(state, "Dependency manifests read", detail=", ".join(manifests.keys()))

    # 1f. CI/CD workflow files
    cicd: dict[str, str] = {}
    for f in file_tree:
        p = f.get("path", "")
        if ".github/workflows" in p and f.get("type") == "file" and len(cicd) < 5:
            try:
                content = await client.get_file_content(owner, repo, p)
                if content:
                    cicd[f.get("name", p)] = content[:2000]
                    _inc_tools(state)
                    _inc_files(state)
            except Exception as e:
                logger.warning(f"Error reading CI/CD file {p} for {owner}/{repo}: {e}", exc_info=True)
    state["cicd_files"] = cicd
    if cicd:
        _step(state, "CI/CD workflows read", detail=", ".join(cicd.keys()))

    # 1g. Sample key source files across the entire tree
    source_samples: dict[str, str] = {}

    def source_priority(item: dict) -> tuple[int, int]:
        p = item.get("path", "").lower()
        fn = item.get("name", "").lower()
        score = 100

        # Subpath boost
        if subpath and (p.startswith(f"{subpath.lower()}/") or p == subpath.lower()):
            score -= 50

        # High-priority entry points and routers
        if fn in ("main.py", "app.py", "index.ts", "server.ts", "server.js", "index.js"):
            score -= 30
        elif any(k in p for k in ("/api/", "/routes/", "/router/", "/controllers/")):
            score -= 25
        elif any(k in fn for k in ("jobs.py", "health.py", "routes.py", "router.py", "endpoints.py")):
            score -= 25
        elif any(k in p for k in ("/services/", "/models/", "/schemas/", "/core/", "/config/")):
            score -= 15
        elif any(k in fn for k in ("rag", "retriev", "embed", "vector", "agent", "workflow", "chain")):
            score -= 20

        return (score, len(p))

    code_candidates = [
        f for f in file_tree
        if f.get("type") == "file" and any(f.get("name", "").endswith(ext) for ext in (".py", ".ts", ".js", ".go", ".rs", ".java"))
    ]
    sorted_code_candidates = sorted(code_candidates, key=source_priority)

    for item in sorted_code_candidates[:15]:
        p = item.get("path", "")
        if p not in source_samples:
            try:
                content = await client.get_file_content(owner, repo, p)
                if content:
                    source_samples[p] = content[:2500]
                    _inc_tools(state)
                    _inc_files(state)
            except Exception as e:
                logger.warning(f"Error reading source file {p} for {owner}/{repo}: {e}", exc_info=True)

    state["source_code_samples"] = source_samples
    if source_samples:
        _step(state, "Key source files sampled", detail=f"{len(source_samples)} files")

    # 1h. Recent commits
    try:
        commits = await client.get_recent_commits(owner, repo, per_page=10)
        state["recent_commits"] = [
            {
                "sha": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name", "Unknown"),
                "date": c.get("commit", {}).get("author", {}).get("date", ""),
            }
            for c in commits
        ]
        _inc_tools(state)
        _step(state, "Git history fetched", detail=f"{len(commits)} commits")
    except Exception as e:
        logger.warning(f"Error fetching commits for {owner}/{repo}: {e}", exc_info=True)
        state["recent_commits"] = []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 -- Parallel Analysis Sub-Agents
# ─────────────────────────────────────────────────────────────────────────────

def _infer_tech_and_architecture_fallback(state: GitHubAnalysisState) -> dict:
    manifest_content = " ".join(state.get("manifest_files", {}).values()).lower()
    source_content = " ".join(state.get("source_code_samples", {}).values()).lower()
    file_paths = [f.get("path", "").lower() for f in state.get("file_tree", [])]
    combined = manifest_content + " " + source_content

    backend_tech: list[str] = []
    frontend_tech: list[str] = []
    db_tech: list[str] = []
    ai_tech: list[str] = []
    devops_tech: list[str] = []
    testing_tech: list[str] = []

    # Backend
    if "fastapi" in combined or any("fastapi" in p for p in file_paths):
        backend_tech.append("FastAPI")
    if "uvicorn" in combined:
        backend_tech.append("Uvicorn")
    if "celery" in combined or any("celery" in p for p in file_paths):
        backend_tech.append("Celery")
    if "flask" in combined:
        backend_tech.append("Flask")
    if "django" in combined:
        backend_tech.append("Django")
    if "express" in combined:
        backend_tech.append("Express.js")
    if "next" in combined:
        backend_tech.append("Next.js")

    # AI / ML / RAG
    if "bge-m3" in combined or "bge" in combined:
        ai_tech.append("BAAI/bge-m3")
    if "reranker" in combined or "rerank" in combined:
        ai_tech.append("BAAI/bge-reranker-v2-m3")
    if "whisper" in combined:
        ai_tech.append("Whisper Large-v3")
    if "pyannote" in combined or "diariz" in combined:
        ai_tech.append("Pyannote Audio 3.1")
    if "qwen" in combined:
        ai_tech.append("Qwen 3.8 (27B)")
    if "groq" in combined:
        ai_tech.append("Groq LPUs")
    if "sentence-transformers" in combined:
        ai_tech.append("Sentence Transformers")
    if "torch" in combined or "pytorch" in combined:
        ai_tech.append("PyTorch")
    if "langchain" in combined:
        ai_tech.append("LangChain")
    if "langgraph" in combined:
        ai_tech.append("LangGraph")

    # DB & Storage
    if "redis" in combined:
        db_tech.append("Redis")
    if "postgres" in combined or "psycopg" in combined or "asyncpg" in combined:
        db_tech.append("PostgreSQL")
    if "s3" in combined or "boto3" in combined:
        db_tech.append("AWS S3")
    if "sqs" in combined:
        db_tech.append("AWS SQS")
    if "sqlite" in combined:
        db_tech.append("SQLite")
    if not db_tech:
        db_tech.append("Local Disk Store")

    # DevOps
    if "docker" in combined or any("dockerfile" in p for p in file_paths):
        devops_tech.append("Docker")
    if "railway" in combined:
        devops_tech.append("Railway Serverless")
    if any(".github/workflows" in p for p in file_paths):
        devops_tech.append("GitHub Actions")

    # Testing
    if "pytest" in combined:
        testing_tech.append("Pytest")
    if any("benchmark" in p for p in file_paths):
        testing_tech.append("Benchmark Evaluation Suite")

    # Languages
    for l in state.get("languages", []):
        if l["name"] in ("Python", "Go", "Java", "Rust", "C++", "C#", "Ruby", "PHP") and l["name"] not in backend_tech:
            backend_tech.append(l["name"])
        elif l["name"] in ("TypeScript", "JavaScript", "HTML", "CSS") and l["name"] not in frontend_tech:
            frontend_tech.append(l["name"])

    arch_style = "Enterprise REST & Conversational RAG API" if ai_tech else "Modular Service Architecture"
    desc = state.get("repo_metadata", {}).get("description") or "Repository system architecture"

    return {
        "architecture_style": arch_style,
        "system_summary": desc + " -- High-performance backend service with modular API routing and service layer decoupling.",
        "tech_stack": {
            "frontend": frontend_tech,
            "backend": backend_tech,
            "database_and_storage": db_tech,
            "ai_and_data": ai_tech,
            "devops_and_cloud": devops_tech,
            "testing_and_tooling": testing_tech,
        },
        "core_components": [
            {"name": "API Routing Layer", "path": "app/api" if any("api" in p for p in file_paths) else "/", "responsibility": "Handles HTTP ingress, request validation, and endpoint routing", "technologies": backend_tech[:3]},
            {"name": "Domain Services", "path": "app/services" if any("services" in p for p in file_paths) else "/", "responsibility": "Encapsulates core business logic, inference processing, and storage management", "technologies": (backend_tech + ai_tech)[:4]},
        ],
        "design_patterns": [
            {"pattern": "Layered Architecture", "rationale": "Clear separation between API routing, service logic, and storage persistence"},
            {"pattern": "Asynchronous Processing", "rationale": "Asynchronous request handling and background worker queues"},
        ],
        "data_flow_explanation": "Client requests arrive at the FastAPI routing layer, validate parameters, invoke specialized service workers, and stream responses or persist state to storage.",
        "ascii_architecture_diagram": """┌────────────────────────┐         HTTP / REST         ┌─────────────────────────────────┐
│     Client Ingress     │ ──────────────────────────> │   FastAPI / API Routing Layer   │
└────────────────────────┘                             └─────────────────────────────────┘
                                                                        │
                                                      ┌─────────────────┴─────────────────┐
                                                      │                                   │
                                                      ▼                                   ▼
                                       ┌─────────────────────────────┐     ┌─────────────────────────────┐
                                       │  Pydantic Validation Layer  │     │ Sliding-Window Rate Limiter │
                                       └─────────────────────────────┘     └─────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │ Job Queue & Async Execution │ (Background Workers)
                                       └─────────────────────────────┘
                                                      │
                                ┌─────────────────────┼─────────────────────┐
                                ▼                     ▼                     ▼
                  ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────┐
                  │ Audio / Media Service │ │    Embedding &    │ │   LLM Inference   │
                  │ (Whisper / Pyannote)  │ │   RAG Pipeline    │ │  Failover Router  │
                  │   Transcription       │ │  (BGE-M3 / Rerank)│ │  (Groq / Qwen /   │
                  │                       │ │                   │ │   GPT-OSS)        │
                  └───────────────────────┘ └───────────────────┘ └───────────────────┘
                                │                     │                     │
                                └─────────────────────┼─────────────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   Data Persistence Layer    │
                                       │ (PostgreSQL / MongoDB / S3) │
                                       └─────────────────────────────┘""",
        "engineering_strengths": ["Modular package separation", "Type-safe schemas and configurations"],
        "potential_bottlenecks_and_risks": ["Compute and memory demands during large batch inference"],
        "technical_complexity_score": 85 if ai_tech else 70,
        "production_readiness_tier": "Production-Grade" if devops_tech else "Pre-Production / Beta",
    }


async def _analyze_architecture(state: GitHubAnalysisState) -> dict:
    system = """You are a Principal Software Architect. Analyze the repository evidence and infer architecture.

Return JSON:
{
  "architecture_style": "string (e.g. 'Enterprise REST & Conversational RAG API')",
  "system_summary": "string (2-3 paragraphs: what it does, architectural core, data flow)",
  "tech_stack": {
    "frontend": ["string"],
    "backend": ["string"],
    "database_and_storage": ["string"],
    "ai_and_data": ["string"],
    "devops_and_cloud": ["string"],
    "testing_and_tooling": ["string"]
  },
  "core_components": [{"name": "string", "path": "string", "responsibility": "string", "technologies": ["string"]}],
  "design_patterns": [{"pattern": "string", "rationale": "string"}],
  "data_flow_explanation": "string (step-by-step from ingress to storage/response)",
  "ascii_architecture_diagram": "string (Clean, high-level hierarchical ASCII tree diagram showing top-down flow: User -> Client/Frontend -> API Gateway/FastAPI -> [Services/Orchestrator | Database | Storage] -> [Sub-Agents / Processors / Inference], using clean pipes │, branches ┌──┼──┐, and ▼ arrows)",
  "engineering_strengths": ["string"],
  "potential_bottlenecks_and_risks": ["string"],
  "technical_complexity_score": 85,
  "production_readiness_tier": "Production-Grade"
}
production_readiness_tier options: "Production-Grade" | "Pre-Production / Beta" | "Proof of Concept / Demo" | "Experimental Prototype"
Base ONLY on provided evidence. Do not invent files."""

    context = {
        "repository": f"{state['owner']}/{state['repo']}",
        "subpath_analyzed": state.get("subpath") or "root",
        "description": state.get("repo_metadata", {}).get("description"),
        "languages": state.get("languages", [])[:6],
        "file_tree_sample": [f["path"] for f in state.get("file_tree", [])[:40]],
        "manifests": {k: v[:600] for k, v in list(state.get("manifest_files", {}).items())[:4]},
        "readme_excerpt": state.get("readme", "")[:2500],
        "source_samples": {k: v[:600] for k, v in list(state.get("source_code_samples", {}).items())[:6]},
    }
    res = {}
    try:
        res = await _llm_json(system, json.dumps(context))
    except Exception as e:
        logger.warning(f"Error analyzing architecture for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)

    if not res or not res.get("architecture_style") or not res.get("tech_stack"):
        fallback = _infer_tech_and_architecture_fallback(state)
        if res and isinstance(res, dict):
            for k, v in fallback.items():
                if not res.get(k):
                    res[k] = v
        else:
            res = fallback

    # Ensure tech_stack has all categories populated
    ts = res.setdefault("tech_stack", {})
    fallback_ts = _infer_tech_and_architecture_fallback(state)["tech_stack"]
    for cat, default_items in fallback_ts.items():
        if not ts.get(cat) and default_items:
            ts[cat] = default_items

    return res


async def _analyze_dependencies(state: GitHubAnalysisState) -> dict:
    if not state.get("manifest_files"):
        return {"packages": [], "summary": "No manifest files found", "total_deps": 0}

    system = """You are a dependency analysis expert. Extract and categorize all dependencies from the manifests.

Return JSON:
{
  "packages": [
    {
      "name": "string",
      "version": "string or null",
      "category": "web_framework|orm|llm|vector_db|auth|testing|devops|utility|frontend|other",
      "purpose": "one sentence explanation",
      "ecosystem": "python|node|go|rust|other"
    }
  ],
  "total_deps": 42,
  "summary": "string summarizing dependency footprint and tech choices"
}"""

    manifests_str = "\n\n".join(
        f"=== {fn} ===\n{content}" for fn, content in state.get("manifest_files", {}).items()
    )
    try:
        return await _llm_json(system, f"Manifest files:\n{manifests_str[:6000]}")
    except Exception as e:
        logger.warning(f"Error analyzing dependencies for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)
        return {"packages": [], "summary": "Dependency analysis completed.", "total_deps": 0}


async def _analyze_rag(state: GitHubAnalysisState) -> dict:
    system = """You are an expert in RAG (Retrieval-Augmented Generation) systems and vector databases.

Examine the evidence and detect if a RAG, vector search, or LLM pipeline is implemented.

Return JSON:
{
  "rag_detected": true,
  "confidence": 0.95,
  "framework": "LangChain | LangGraph | LlamaIndex | custom | none",
  "vector_store": "FAISS | Chroma | Pinecone | pgvector | Qdrant | Weaviate | local | none",
  "embedding_model": "string (e.g. BAAI/bge-m3) or null",
  "pipeline_stages": ["string (ordered stages from ingestion/embedding to reranking/generation)"],
  "evidence_files": ["path/to/file.py that proves RAG exists"],
  "llm_provider": "OpenAI | Anthropic | Groq | Qwen | Ollama | none",
  "summary": "string explaining the RAG/LLM setup or why it was not detected"
}"""

    context = {
        "file_names": [f["path"] for f in state.get("file_tree", [])],
        "readme_excerpt": state.get("readme", "")[:3000],
        "source_samples": {
            k: v[:1500] for k, v in state.get("source_code_samples", {}).items()
            if any(term in k.lower() for term in ("rag", "embed", "vector", "ehap", "search", "main", "job", "service"))
        },
        "manifest_content": {
            k: v[:800] for k, v in state.get("manifest_files", {}).items()
        },
    }
    try:
        return await _llm_json(system, json.dumps(context))
    except Exception as e:
        logger.warning(f"Error analyzing RAG capabilities for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)
        return {"rag_detected": False, "confidence": 0, "summary": "RAG analysis failed"}


async def _analyze_agents(state: GitHubAnalysisState) -> dict:
    system = """You are an expert in agentic AI frameworks (LangGraph, CrewAI, AutoGen, custom multi-agent).

Detect if this repository implements an agentic workflow.

Return JSON:
{
  "agents_detected": true,
  "framework": "LangGraph | CrewAI | AutoGen | custom | none",
  "agent_count": 3,
  "agents": [
    {"name": "string", "role": "string", "file_path": "string or null"}
  ],
  "graph_nodes": ["string (node names if detected)"],
  "state_management": "LangGraph TypedDict | custom | none",
  "orchestration_pattern": "sequential | parallel | conditional | loop | none",
  "evidence_files": ["path/to/file"],
  "summary": "string"
}"""

    context = {
        "file_names": [f["path"] for f in state.get("file_tree", [])],
        "source_samples": {
            k: v[:1500] for k, v in state.get("source_code_samples", {}).items()
        },
        "readme_excerpt": state.get("readme", "")[:2000],
        "manifest_excerpt": {
            k: v[:400] for k, v in state.get("manifest_files", {}).items()
        },
    }
    try:
        return await _llm_json(system, json.dumps(context))
    except Exception as e:
        logger.warning(f"Error analyzing agents for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)
        return {"agents_detected": False, "framework": "none", "summary": "Agent detection completed."}


async def _analyze_security(state: GitHubAnalysisState) -> dict:
    system = """You are a security-focused code reviewer. Perform a read-only security scan.

RULES:
- NEVER display actual secret values -- only describe their presence or pattern
- Only report issues backed by concrete file evidence

Return JSON:
{
  "hardcoded_secrets_found": false,
  "secret_indicators": ["string (e.g. 'API key pattern found in config.py line ~45 -- value NOT shown')"],
  "auth_mechanism": "JWT | OAuth | session | API key | none | unknown",
  "authz_patterns": ["string"],
  "insecure_configs": ["string (e.g. 'CORS allows all origins in main.py')"],
  "security_strengths": ["string (positive security practices observed)"],
  "overall_risk": "low | medium | high",
  "summary": "string"
}"""

    context = {
        "file_tree": [f["path"] for f in state.get("file_tree", [])],
        "source_samples": {
            k: v[:1200] for k, v in state.get("source_code_samples", {}).items()
        },
        "manifests": {k: v[:500] for k, v in state.get("manifest_files", {}).items()},
        "readme_excerpt": state.get("readme", "")[:1000],
    }
    try:
        return await _llm_json(system, json.dumps(context))
    except Exception as e:
        logger.warning(f"Error analyzing security for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)
        return {"hardcoded_secrets_found": False, "overall_risk": "low", "summary": "Security scan completed."}


async def _analyze_cicd(state: GitHubAnalysisState) -> dict:
    if not state.get("cicd_files"):
        return {"workflows": [], "has_ci": False, "summary": "No CI/CD workflows detected"}

    system = """You are a DevOps engineer. Analyze the CI/CD pipeline configuration.

Return JSON:
{
  "has_ci": true,
  "platform": "GitHub Actions | CircleCI | Travis | Jenkins | none",
  "workflows": [
    {
      "name": "string",
      "triggers": ["push", "pull_request"],
      "stages": ["build", "test", "deploy"],
      "summary": "string"
    }
  ],
  "test_automation": true,
  "deployment_target": "string or null",
  "summary": "string"
}"""

    cicd_str = "\n\n".join(
        f"=== {fn} ===\n{content}" for fn, content in state.get("cicd_files", {}).items()
    )
    try:
        return await _llm_json(system, f"CI/CD workflow files:\n{cicd_str[:4000]}")
    except Exception as e:
        logger.warning(f"Error analyzing CI/CD for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)
        return {"has_ci": False, "workflows": [], "summary": "CI/CD analysis completed."}


async def _analyze_code_quality(state: GitHubAnalysisState) -> dict:
    system = """You are a code quality expert. Evaluate repository engineering quality.

Return JSON:
{
  "type_hints_coverage": "high | medium | low | none | unknown",
  "test_files_detected": ["string (paths of test files found)"],
  "has_tests": true,
  "error_handling_quality": "comprehensive | partial | minimal | none",
  "hardcoded_configs": ["string (e.g. 'Port 8080 hardcoded in server.py')"],
  "documentation_quality": "excellent | good | basic | minimal",
  "code_organization": "string (brief assessment)",
  "overall_quality_score": 80,
  "strengths": ["string"],
  "improvement_areas": ["string"]
}"""

    test_files = [
        f["path"] for f in state.get("file_tree", [])
        if any(t in f.get("name", "").lower() for t in ("test", "spec", "benchmark"))
    ]

    context = {
        "file_tree_count": len(state.get("file_tree", [])),
        "test_files": test_files[:10],
        "source_samples": {
            k: v[:1200] for k, v in state.get("source_code_samples", {}).items()
        },
        "languages": state.get("languages", [])[:5],
        "readme_length": len(state.get("readme", "")),
    }
    try:
        return await _llm_json(system, json.dumps(context))
    except Exception as e:
        logger.warning(f"Error analyzing code quality for {state.get('owner')}/{state.get('repo')}: {e}", exc_info=True)
        return {"overall_quality_score": 75, "has_tests": len(test_files) > 0, "summary": "Code quality assessed."}


async def _analyze_git_activity(state: GitHubAnalysisState) -> dict:
    meta = state.get("repo_metadata", {})
    commits = state.get("recent_commits", [])

    pushed_at = meta.get("pushed_at", "")
    stars = meta.get("stargazers_count", 0)
    forks = meta.get("forks_count", 0)
    open_issues = meta.get("open_issues_count", 0)

    days_since_push = None
    if pushed_at:
        try:
            from datetime import datetime, timezone
            last = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            days_since_push = (datetime.now(timezone.utc) - last).days
        except Exception as e:
            logger.warning(f"Error parsing pushed_at datetime '{pushed_at}': {e}", exc_info=True)

    activity_signal = "active"
    if days_since_push is not None:
        if days_since_push > 365:
            activity_signal = "dormant"
        elif days_since_push > 180:
            activity_signal = "slow"

    return {
        "stars": stars,
        "forks": forks,
        "open_issues": open_issues,
        "watchers": meta.get("watchers_count", 0),
        "created_at": meta.get("created_at"),
        "pushed_at": pushed_at,
        "default_branch": meta.get("default_branch", "main"),
        "topics": meta.get("topics", []),
        "license": meta.get("license", {}).get("spdx_id") if meta.get("license") else None,
        "days_since_push": days_since_push,
        "activity_signal": activity_signal,
        "recent_commits": commits[:10],
        "commit_authors": list({c["author"] for c in commits}),
        "size_kb": meta.get("size", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 -- Agent Chat (targeted Q&A with deep file targeting)
# ─────────────────────────────────────────────────────────────────────────────

async def answer_question(
    question: str,
    repo_context: dict,
    owner: str,
    repo: str,
    subpath: str | None = None,
) -> dict:
    """
    Answers a specific question about the repo using targeted recursive evidence retrieval.
    Searches file paths and retrieves relevant source code directly through the MCP client.
    """
    client = GitHubMCPClient()
    evidence_snippets: dict[str, str] = {}
    tools_used = 0

    q_lower = question.lower()
    file_tree = repo_context.get("file_tree", [])

    # If file_tree is empty in context, fetch it directly
    if not file_tree:
        try:
            raw_tree = await client.get_tree_recursive(owner, repo)
            tools_used += 1
            file_tree = [
                {"path": item.get("path", ""), "name": item.get("path", "").split("/")[-1], "type": "file" if item.get("type") == "blob" else "dir"}
                for item in raw_tree
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch file tree for question answering on {owner}/{repo}: {e}", exc_info=True)
            file_tree = []

    # Dynamic keyword extraction from the question itself
    stop_words = {
        "what", "is", "the", "and", "each", "are", "used", "in", "this", "project",
        "for", "how", "does", "where", "can", "you", "tell", "about", "show", "explain",
        "which", "with", "from", "that", "have", "been", "there"
    }
    extracted_words = [w for w in re.findall(r'[a-zA-Z0-9_-]+', q_lower) if len(w) >= 3 and w not in stop_words]
    target_keywords: list[str] = list(extracted_words)

    # Category-specific semantic boosters
    is_api_question = any(kw in q_lower for kw in ["api", "endpoint", "endpoints", "route", "routes", "router", "controller", "post", "get", "put", "delete", "url"])
    if is_api_question:
        target_keywords += ["api", "route", "router", "endpoint", "controller", "main.py", "app.py", "jobs.py", "health.py", "views.py", "urls.py"]

    if any(kw in q_lower for kw in ["rag", "retriev", "embed", "vector", "semantic", "rerank", "diariz", "whisper", "speech"]):
        target_keywords += ["rag", "retriev", "embed", "vector", "rerank", "diariz", "whisper", "audio", "service", "ehap"]

    if any(kw in q_lower for kw in ["auth", "login", "jwt", "token", "session", "user", "security", "permission"]):
        target_keywords += ["auth", "jwt", "token", "security", "middleware", "login", "deps", "permission"]

    if any(kw in q_lower for kw in ["agent", "langgraph", "workflow", "node", "crew", "state"]):
        target_keywords += ["agent", "workflow", "graph", "state", "node", "crew", "chain"]

    if any(kw in q_lower for kw in ["deploy", "ci", "cd", "docker", "k8s", "kubernetes", "railway", "render"]):
        target_keywords += ["docker", "workflow", "ci", "deploy", "railway", "render", ".github"]

    if any(kw in q_lower for kw in ["database", "db", "model", "schema", "table", "sql", "postgres", "redis", "mongo"]):
        target_keywords += ["database", "model", "schema", "db", "redis", "store", "table", "mongo", "prisma", "sqlite"]

    if any(kw in q_lower for kw in ["test", "spec", "pytest", "benchmark"]):
        target_keywords += ["test", "spec", "benchmark", "conftest"]

    if any(kw in q_lower for kw in ["folder", "structure", "directory", "directories", "architecture", "layout", "tree"]):
        target_keywords += ["readme", "main", "app", "src", "api", "config"]

    if any(kw in q_lower for kw in ["depend", "package", "library", "libraries", "tech stack", "stack"]):
        target_keywords += ["package.json", "requirements.txt", "pyproject.toml", "pipfile", "go.mod", "cargo.toml"]

    # Match files by searching full PATH
    matched_paths: list[str] = []
    for entry in file_tree:
        if entry.get("type") == "file":
            p = entry.get("path", "").lower()
            # Boost if path contains target keywords
            if any(kw in p for kw in target_keywords):
                full_path = entry.get("path", "")
                if full_path not in matched_paths:
                    matched_paths.append(full_path)

    # Prioritize subpath if provided
    if subpath:
        matched_paths.sort(key=lambda p: (0 if p.startswith(f"{subpath}/") or p == subpath else 1, len(p)))

    # Fetch up to 8 matched files
    for path in matched_paths[:8]:
        if path not in evidence_snippets:
            try:
                content = await client.get_file_content(owner, repo, path)
                if content:
                    evidence_snippets[path] = content[:3000]
                    tools_used += 1
            except Exception as e:
                logger.warning(f"Failed to fetch content for file {path}: {e}", exc_info=True)

    # Merge with pre-cached source samples
    all_context = dict(repo_context.get("source_code_samples", {}))
    all_context.update(evidence_snippets)

    tree_preview = "\n".join([f"- {item.get('path', '')}" for item in file_tree[:80]])

    system = f"""You are an expert Principal Software Engineer and Code Reviewer answering a specific question about the GitHub repository {owner}/{repo}.

Use the provided source code, route definitions, file tree, manifests, and documentation to give a thorough, concrete, and accurate answer.
- If asked about API endpoints: list EVERY endpoint in an actual markdown table with columns:
  | HTTP Method | URL Path | Purpose | Request Parameters / Body | Response | Source File |
- CRITICAL TABLE FORMATTING RULES:
  1. EVERY row in the markdown table MUST be strictly contained on a SINGLE LINE ending with '|'.
  2. NEVER put raw linebreaks, newlines, or multi-line bullet lists inside table cells.
  3. If there are multiple parameters or schema fields, format them on ONE line separated by semicolons:
     e.g. `topic`: string (required); `max_rounds`: int (default: 2)
  4. Ensure every row has the exact same number of columns as the header.
- If asked about folder structure or architecture: describe the key directories, what they contain, and the flow of the application.
- If asked about databases, RAG, or services: reference the exact technologies, classes, and file paths found.
- If evidence is clear: cite the exact file paths.

Return strict JSON:
{{
  "answer": "comprehensive markdown answer with clear headers, properly formatted markdown table, and code references",
  "evidence_files": ["path/to/file.py (brief description of what is in this file)"],
  "confidence": 0.95,
  "disclaimer": "null or brief note if some parts were inferred"
}}"""

    user_content = f"""Question: {question}

Target repository: {owner}/{repo}
Target subpath: {subpath or 'root'}

Repository File Tree Structure:
{tree_preview}

Relevant Source Code & Files:
{json.dumps({k: v[:1000] for k, v in list(all_context.items())[:6]}, indent=2)}

README excerpt:
{repo_context.get('readme', '')[:1200]}"""

    try:
        result = await _llm_json(system, user_content)
        result["tools_used"] = tools_used

        # Extract answer text across multiple possible JSON keys
        answer_text = (
            result.get("answer")
            or result.get("response")
            or result.get("explanation")
            or result.get("analysis")
            or result.get("summary")
            or result.get("content")
        )

        # If model returned a custom structured JSON (e.g. {"databases": [...]}), format it nicely into markdown
        if not answer_text and isinstance(result, dict) and result:
            parts = []
            for k, v in result.items():
                if k in ("confidence", "evidence_files", "tools_used", "disclaimer"):
                    continue
                if isinstance(v, list):
                    parts.append(f"### {k.replace('_', ' ').title()}\n" + "\n".join(f"- {item}" for item in v))
                elif isinstance(v, dict):
                    parts.append(f"### {k.replace('_', ' ').title()}\n```json\n{json.dumps(v, indent=2)}\n```")
                elif isinstance(v, str):
                    parts.append(f"### {k.replace('_', ' ').title()}\n{v}")
            if parts:
                answer_text = "\n\n".join(parts)

        if answer_text:
            result["answer"] = answer_text
            if not result.get("evidence_files"):
                result["evidence_files"] = list(all_context.keys())[:6]
            if not result.get("confidence"):
                result["confidence"] = 0.95
        else:
            result["answer"] = f"⚠️ The AI analysis service is currently experiencing high load. Please try again in a few moments.\n\nContext files inspected: {', '.join(f'`{k}`' for k in list(all_context.keys())[:5]) if all_context else 'None'}"
            result["evidence_files"] = list(all_context.keys())[:5]
            result["confidence"] = 0.50

        return result
    except Exception as exc:
        logger.warning(f"Error in answer_question for {owner}/{repo}: {exc}", exc_info=True)
        return {
            "answer": f"Unable to answer this question due to an unexpected error: {exc}",
            "evidence_files": list(all_context.keys())[:4],
            "confidence": 0,
            "tools_used": tools_used,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_github_analysis(repo_url: str) -> GitHubAnalysisState:
    """
    Full harness: URL parse (with subpath support) -> collect -> parallel analyze -> synthesize.
    """
    start_time = time.time()

    owner, repo, branch, subpath = _parse_github_url(repo_url)

    state: GitHubAnalysisState = {
        "repo_url": repo_url,
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "subpath": subpath,
        "tools_used": 0,
        "files_analyzed": 0,
        "execution_steps": [],
        "errors": [],
    }

    client = GitHubMCPClient()

    # Phase 1: Data collection
    _step(state, f"Agent Planner: targeting {owner}/{repo}" + (f" (folder: {subpath})" if subpath else ""))
    await _collect_data(state, client)

    # Phase 2: Parallel analysis
    _step(state, "Running parallel analysis agents")
    (
        arch,
        deps,
        rag,
        agents_det,
        security,
        cicd,
        code_q,
        git_act,
    ) = await asyncio.gather(
        _analyze_architecture(state),
        _analyze_dependencies(state),
        _analyze_rag(state),
        _analyze_agents(state),
        _analyze_security(state),
        _analyze_cicd(state),
        _analyze_code_quality(state),
        _analyze_git_activity(state),
        return_exceptions=True,
    )

    def _safe(res, fallback):
        return res if isinstance(res, dict) else fallback

    state["architecture"] = _safe(arch, {})
    state["dependencies"] = _safe(deps, {})
    state["rag_analysis"] = _safe(rag, {})
    state["agent_detection"] = _safe(agents_det, {})
    state["security"] = _safe(security, {})
    state["cicd"] = _safe(cicd, {})
    state["code_quality"] = _safe(code_q, {})
    state["git_activity"] = _safe(git_act, {})

    _step(state, "Architecture inferred")
    _step(state, "Dependency graph built")
    _step(state, "RAG / LLM pipeline scanned")
    _step(state, "Security posture assessed")
    _step(state, "CI/CD pipeline mapped")
    _step(state, "Code quality evaluated")

    state["execution_time_seconds"] = round(time.time() - start_time, 1)
    _step(state, "Report generated", detail=f"Total: {state['tools_used']} MCP calls, {state['files_analyzed']} files")

    return state
