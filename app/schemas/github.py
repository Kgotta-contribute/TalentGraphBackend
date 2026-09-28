from pydantic import BaseModel
from typing import Literal

class GitHubEvidence(BaseModel):
    skill: str
    verified: bool
    repo: str
    file_path: str | None = None
    evidence: str
    confidence: float

class GitHubAnalysis(BaseModel):
    candidate_id: str
    github_url: str
    repos_analyzed: list[str]
    verified_skills: list[str]
    unverified_skills: list[str]
    activity_signal: Literal["active", "dormant", "forks_only", "unavailable"]
    evidence: list[GitHubEvidence]
    summary: str
