from typing import Any
from pydantic import BaseModel, Field

class ExperienceItem(BaseModel):
    company: str | None = ""
    title: str | None = ""
    start: str | None = ""
    end: str | None = None
    description: str | None = ""
    tech_tags: list[str] = Field(default_factory=list)

class ProjectItem(BaseModel):
    name: str | None = ""
    description: str | None = ""
    tech_tags: list[str] = Field(default_factory=list)
    github_url: str | None = None

class EducationItem(BaseModel):
    degree: str | None = ""
    institution: str | None = ""
    graduation_year: Any = None

class CandidateProfile(BaseModel):
    full_name: str | None = "Candidate"
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    current_title: str | None = None
    years_experience: Any = 0
    summary: str | None = ""
    skills: list[str] = Field(default_factory=list)
    experience_history: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    github_url: str | None = None
    linkedin_url: str | None = None
