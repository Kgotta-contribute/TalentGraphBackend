from pydantic import BaseModel, field_validator
from typing import Literal

VALID_SOURCES = {"RESUME", "PROJECT", "GITHUB", "EDUCATION"}
VALID_STATUSES = {"MATCHED", "PARTIAL", "MISSING", "UNKNOWN"}

class EvidenceItem(BaseModel):
    requirement: str
    status: Literal["MATCHED", "PARTIAL", "MISSING", "UNKNOWN"] = "UNKNOWN"
    source: Literal["RESUME", "PROJECT", "GITHUB", "EDUCATION"] = "RESUME"
    evidence: str = ""
    confidence: float = 0.0

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, v: str) -> str:
        if not isinstance(v, str):
            return "RESUME"
        v_upper = v.upper().strip()
        if v_upper in VALID_SOURCES:
            return v_upper
        if any(term in v_upper for term in ["EXP", "WORK", "JOB", "ROLE", "HISTORY"]):
            return "RESUME"
        if "PROJ" in v_upper:
            return "PROJECT"
        if "GIT" in v_upper:
            return "GITHUB"
        if any(term in v_upper for term in ["EDU", "DEGREE", "COLLEGE", "UNIV", "ACADEMIC"]):
            return "EDUCATION"
        return "RESUME"

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: str) -> str:
        if not isinstance(v, str):
            return "UNKNOWN"
        v_upper = v.upper().strip()
        if v_upper in VALID_STATUSES:
            return v_upper
        if any(term in v_upper for term in ["MATCH", "YES", "TRUE", "MET", "FOUND"]):
            return "MATCHED"
        if "PART" in v_upper:
            return "PARTIAL"
        if any(term in v_upper for term in ["MISS", "NO", "GAP", "NOT"]):
            return "MISSING"
        return "UNKNOWN"

class VerificationResult(BaseModel):
    candidate_id: str = ""
    matched_required: list[str] = []
    matched_preferred: list[str] = []
    required_gaps: list[str] = []
    preferred_gaps: list[str] = []
    partial_matches: list[str] = []
    evidence: list[EvidenceItem] = []
    tech_coverage_pct: float = 0.0

class ScoringFactors(BaseModel):
    technical_skills: float      # raw 0-100
    experience_tenure: float     # raw 0-100
    jd_similarity: float         # raw 0-100 cosine similarity
    project_relevance: float     # raw 0-100
    education_certs: float       # raw 0-100

class ScoringWeights(BaseModel):
    technical_skills: float = 0.40
    experience_tenure: float = 0.25
    jd_similarity: float = 0.20
    project_relevance: float = 0.10
    education_certs: float = 0.05
    
    def validate_sum(self):
        total = sum([self.technical_skills, self.experience_tenure, 
                     self.jd_similarity, self.project_relevance, self.education_certs])
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

TIER_THRESHOLDS = [
    (90, "Strongly Recommended"),
    (75, "Recommended"),
    (60, "Interview Candidate"),
    (40, "Weak Match"),
    (0, "Not Recommended"),
]

class ScoringResult(BaseModel):
    candidate_id: str
    mandate_id: str
    raw_factors: ScoringFactors
    weights: ScoringWeights
    weighted_contributions: dict[str, float]
    final_score: float
    tier: str
