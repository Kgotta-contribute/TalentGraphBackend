from pydantic import BaseModel, Field

class JobRequirements(BaseModel):
    role: str = "Software Engineer"
    company: str | None = None
    experience_target_years: int = 3
    education_criteria: str = "Bachelor's degree in Computer Science, Engineering, or equivalent practical experience"
    mandatory_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
