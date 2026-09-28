from pydantic import BaseModel

class InterviewQuestion(BaseModel):
    focus_area: str
    question: str
    rationale: str
    keywords: list[str]

class RecruitmentDossier(BaseModel):
    candidate_id: str
    executive_summary: str
    key_strengths: list[str]
    identified_skill_gaps: list[str]
    risk_factors: list[str]
    ramp_up_considerations: list[str]
    final_verdict: str
    hiring_confidence: float   # 0.0-1.0
    relevant_experience: list[dict]
    relevant_projects: list[dict]
    github_summary: str | None = None
    interview_questions: list[InterviewQuestion]
