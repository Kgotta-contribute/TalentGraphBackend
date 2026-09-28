# -*- coding: utf-8 -*-
"""
Unit tests for Pydantic v2 data contract integrity:
Validates strict schema enforcement for all agent inputs and outputs:
- JobRequirements
- CandidateProfile
- VerificationResult
- ScoringResult
- RecruitmentDossier
"""
import pytest
from pydantic import ValidationError
from app.schemas.job_requirements import JobRequirements
from app.schemas.candidate import (
    CandidateProfile,
    ExperienceItem,
    ProjectItem,
    EducationItem,
)
from app.schemas.evaluation import (
    VerificationResult,
    EvidenceItem,
    ScoringFactors,
    ScoringWeights,
    ScoringResult,
)
from app.schemas.report import RecruitmentDossier, InterviewQuestion


def test_job_requirements_schema_valid():
    """Verify JobRequirements parses valid fields with proper types."""
    data = {
        "role": "Senior Full-Stack Engineer",
        "experience_target_years": 4,
        "education_criteria": "Bachelor's degree in Computer Science or equivalent",
        "mandatory_skills": ["React", "FastAPI", "PostgreSQL"],
        "preferred_skills": ["Docker", "Kubernetes"],
        "soft_skills": ["Leadership", "Teamwork"],
        "responsibilities": ["Design APIs", "Mentor juniors"],
        "domain_tags": ["Full-Stack", "Web"],
    }
    reqs = JobRequirements(**data)
    assert reqs.role == "Senior Full-Stack Engineer"
    assert reqs.experience_target_years == 4
    assert len(reqs.mandatory_skills) == 3
    assert "FastAPI" in reqs.mandatory_skills


def test_job_requirements_schema_validation_error():
    """Verify JobRequirements raises ValidationError on type mismatch."""
    with pytest.raises(ValidationError):
        JobRequirements(experience_target_years="invalid_tenure_string")


def test_candidate_profile_schema_valid():
    """Verify CandidateProfile schema parses complex nested objects."""
    profile = CandidateProfile(
        full_name="Alex Rivera",
        email="alex@example.com",
        years_experience=6.5,
        summary="Experienced engineer specialized in distributed systems",
        skills=["Python", "Go", "Docker", "AWS"],
        experience_history=[
            ExperienceItem(
                company="Acme Corp",
                title="Staff Engineer",
                start="2021-01",
                end="Present",
                description="Architected core payment service",
                tech_tags=["Python", "Kafka"],
            )
        ],
        projects=[
            ProjectItem(
                name="VectorStore-Py",
                description="Fast in-memory vector indexer",
                tech_tags=["Python", "SIMD"],
                github_url="https://github.com/alex/vectorstore-py",
            )
        ],
        education=[
            EducationItem(
                degree="B.S. in Computer Science",
                institution="MIT",
                graduation_year=2018,
            )
        ],
    )
    assert profile.full_name == "Alex Rivera"
    assert profile.years_experience == 6.5
    assert len(profile.experience_history) == 1
    assert profile.experience_history[0].company == "Acme Corp"
    assert len(profile.projects) == 1
    assert profile.projects[0].github_url == "https://github.com/alex/vectorstore-py"


def test_verification_result_schema_valid():
    """Verify VerificationResult parses status and evidence items correctly."""
    evidence = [
        EvidenceItem(
            requirement="FastAPI",
            status="MATCHED",
            source="RESUME",
            evidence="Built RESTful APIs with FastAPI and Pydantic",
            confidence=0.95,
        ),
        EvidenceItem(
            requirement="Kubernetes",
            status="PARTIAL",
            source="PROJECT",
            evidence="Deployed manifests locally using minikube",
            confidence=0.60,
        ),
    ]
    res = VerificationResult(
        candidate_id="cand-12345",
        matched_required=["FastAPI"],
        matched_preferred=[],
        required_gaps=[],
        preferred_gaps=["Kubernetes"],
        partial_matches=["Kubernetes"],
        evidence=evidence,
        tech_coverage_pct=75.0,
    )
    assert res.candidate_id == "cand-12345"
    assert res.tech_coverage_pct == 75.0
    assert len(res.evidence) == 2
    assert res.evidence[0].status == "MATCHED"


def test_recruitment_dossier_schema_valid():
    """Verify RecruitmentDossier data contract and interview probes format."""
    questions = [
        InterviewQuestion(
            focus_area="Database Concurrency",
            question="How do you handle pessimistic locking in PostgreSQL transactions?",
            rationale="Verifies claimed production relational database depth",
            keywords=["ACID", "isolation level", "SELECT FOR UPDATE"],
        )
    ]
    dossier = RecruitmentDossier(
        candidate_id="cand-9999",
        executive_summary="Solid senior backend engineer with deep Python experience.",
        key_strengths=["API architecture", "Database indexing", "Mentorship"],
        identified_skill_gaps=["No direct Kubernetes production experience"],
        risk_factors=["Tenure at previous role was short (<1 year)"],
        ramp_up_considerations=["Allow 2 weeks to get familiar with our Helm charts"],
        final_verdict="Strong candidate for the Senior Backend position.",
        hiring_confidence=0.88,
        relevant_experience=[{"company": "TechCorp", "relevance": "High"}],
        relevant_projects=[{"name": "AuthService", "relevance": "High"}],
        interview_questions=questions,
    )
    assert dossier.candidate_id == "cand-9999"
    assert dossier.hiring_confidence == 0.88
    assert len(dossier.interview_questions) == 1
    assert "ACID" in dossier.interview_questions[0].keywords
