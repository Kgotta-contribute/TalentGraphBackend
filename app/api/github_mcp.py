import re
import json
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.core.deps import get_current_user
from app.agents.github_intelligence_agent import run_github_analysis, answer_question, _parse_github_url

router = APIRouter()


class RepoAnalysisRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL or owner/repo format")


class ChatRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    question: str = Field(..., description="Question to answer about the repository")
    repo_context: dict = Field(default_factory=dict, description="Cached repo context (file_tree, source_samples, readme)")


@router.get("/rate-limits")
async def get_system_rate_limits():
    """
    Returns real-time usage and capacity for Groq LLM (20 RPM)
    and GitHub MCP (20 RPM & 850 RPH) rate limiters.
    """
    from app.core.rate_limiter import groq_rate_limiter, github_rate_limiter
    return {
        "groq": groq_rate_limiter.get_status(),
        "github": github_rate_limiter.get_status(),
    }


@router.post("/analyze-repo")
async def analyze_github_repo(
    body: RepoAnalysisRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    GitHub Intelligence Harness — full multi-step agentic analysis.

    Pipeline:
      GitHub URL → Agent Planner → MCP Tool Calls (repo info, file tree, code reads) →
      Parallel Analysis Agents [Architecture, RAG, Security, CI/CD, Deps, Code Quality] →
      Evidence Store → Structured Report
    """
    try:
        _parse_github_url(body.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        state = await run_github_analysis(body.repo_url)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(e)}"
        )

    # Shape the response for the frontend
    meta = state.get("repo_metadata", {})
    return {
        "repo_url": state.get("repo_url", body.repo_url),
        "subpath": state.get("subpath"),
        "branch": state.get("branch"),
        # Core repo info
        "repo_info": {
            "owner": state["owner"],
            "name": state["repo"],
            "full_name": meta.get("full_name", f"{state['owner']}/{state['repo']}"),
            "html_url": meta.get("html_url", f"https://github.com/{state['owner']}/{state['repo']}"),
            "description": meta.get("description") or "No description provided.",
            "stars": meta.get("stargazers_count", 0),
            "forks": meta.get("forks_count", 0),
            "open_issues": meta.get("open_issues_count", 0),
            "watchers": meta.get("watchers_count", 0),
            "license": meta.get("license", {}).get("spdx_id") if meta.get("license") else "Not specified",
            "default_branch": meta.get("default_branch", "main"),
            "topics": meta.get("topics", []),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "pushed_at": meta.get("pushed_at"),
            "size_kb": meta.get("size", 0),
        },
        # Analysis sections
        "languages": state.get("languages", []),
        "file_tree": state.get("file_tree", []),
        "recent_commits": state.get("recent_commits", []),
        "manifest_files": state.get("manifest_files", {}),
        "cicd_files": state.get("cicd_files", {}),
        # Analysis results from each sub-agent
        "architecture": state.get("architecture", {}),
        "dependencies": state.get("dependencies", {}),
        "rag_analysis": state.get("rag_analysis", {}),
        "agent_detection": state.get("agent_detection", {}),
        "security": state.get("security", {}),
        "cicd_analysis": state.get("cicd", {}),
        "code_quality": state.get("code_quality", {}),
        "git_activity": state.get("git_activity", {}),
        # Cached for chat tab
        "source_code_samples": state.get("source_code_samples", {}),
        "readme": state.get("readme", ""),
        # Observability
        "observability": {
            "tools_used": state.get("tools_used", 0),
            "files_analyzed": state.get("files_analyzed", 0),
            "execution_time_seconds": state.get("execution_time_seconds", 0),
            "steps": state.get("execution_steps", []),
            "errors": state.get("errors", []),
        },
    }


@router.post("/chat")
async def github_repo_chat(
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Agent Chat — answer specific questions about the repository with targeted evidence retrieval.
    The harness fetches only the relevant files to answer the question (not a full repo dump).
    """
    try:
        owner, repo, branch, subpath = _parse_github_url(body.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = await answer_question(
            question=body.question,
            repo_context=body.repo_context,
            owner=owner,
            repo=repo,
            subpath=subpath,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat agent failed: {str(e)}")
