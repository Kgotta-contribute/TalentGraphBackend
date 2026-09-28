# -*- coding: utf-8 -*-
"""
Unit tests for deterministic scoring engine invariants:
- Sum of weights == 1.0 validation
- Factor boundary constraints (0 <= factor <= 100)
- Tier threshold assignments
- End-to-end mathematical reproducibility
"""
import pytest
from app.schemas.evaluation import ScoringWeights, ScoringFactors, TIER_THRESHOLDS, VerificationResult
from app.schemas.candidate import CandidateProfile, ProjectItem
from app.schemas.job_requirements import JobRequirements
from app.ranking.scoring import (
    calculate_technical_score,
    calculate_experience_score,
    assign_tier,
    calculate_final_score,
    DEFAULT_WEIGHTS,
)


def test_weights_sum_validation_success():
    """Verify that default weights sum strictly to 1.0."""
    weights = ScoringWeights()
    weights.validate_sum()
    total = sum([
        weights.technical_skills,
        weights.experience_tenure,
        weights.jd_similarity,
        weights.project_relevance,
        weights.education_certs,
    ])
    assert abs(total - 1.0) < 0.001


def test_weights_sum_validation_failure():
    """Verify that invalid weights that do not sum to 1.0 raise ValueError."""
    invalid_weights = ScoringWeights(
        technical_skills=0.5,
        experience_tenure=0.3,
        jd_similarity=0.1,
        project_relevance=0.05,
        education_certs=0.0,  # Total = 0.95
    )
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        invalid_weights.validate_sum()


def test_calculate_technical_score_boundaries():
    """Test boundary values for technical skills calculation."""
    # Empty mandatory skills fallback
    assert calculate_technical_score([], [], [], []) == 50.0

    # 100% match on both required and preferred
    full_score = calculate_technical_score(
        matched_required=["Python", "FastAPI"],
        mandatory_skills=["Python", "FastAPI"],
        matched_preferred=["Docker"],
        preferred_skills=["Docker"],
    )
    assert full_score == 100.0

    # 0% match on required, 0% on preferred
    zero_score = calculate_technical_score(
        matched_required=[],
        mandatory_skills=["Python", "FastAPI"],
        matched_preferred=[],
        preferred_skills=["Docker"],
    )
    assert zero_score == 0.0

    # Partial match
    partial_score = calculate_technical_score(
        matched_required=["Python"],
        mandatory_skills=["Python", "FastAPI"],  # 50% * 0.8 = 0.40
        matched_preferred=["Docker"],
        preferred_skills=["Docker"],             # 100% * 0.2 = 0.20 -> 60.0
    )
    assert partial_score == pytest.approx(60.0)


def test_calculate_experience_score_boundaries():
    """Test boundary values for candidate tenure calculation."""
    # Target years 0 fallback
    assert calculate_experience_score(candidate_years=5, target_years=0) == 75.0

    # Exact target tenure (ratio = 1.0) -> 85.0
    assert calculate_experience_score(candidate_years=5.0, target_years=5) == 85.0

    # Overqualified / experienced (ratio >= 1.5) -> 100.0
    assert calculate_experience_score(candidate_years=8.0, target_years=5) == 100.0

    # Minimum threshold (ratio = 0.7) -> 60.0
    assert calculate_experience_score(candidate_years=3.5, target_years=5) == 60.0

    # 0 experience
    assert calculate_experience_score(candidate_years=0.0, target_years=5) == 0.0


def test_assign_tier_thresholds():
    """Verify tier assignment across all threshold boundaries."""
    assert assign_tier(95.0) == "Strongly Recommended"
    assert assign_tier(90.0) == "Strongly Recommended"
    assert assign_tier(89.9) == "Recommended"
    assert assign_tier(75.0) == "Recommended"
    assert assign_tier(74.9) == "Interview Candidate"
    assert assign_tier(60.0) == "Interview Candidate"
    assert assign_tier(59.9) == "Weak Match"
    assert assign_tier(40.0) == "Weak Match"
    assert assign_tier(39.9) == "Not Recommended"
    assert assign_tier(0.0) == "Not Recommended"


def test_calculate_final_score_reproducibility():
    """End-to-end test verifying scoring determinism and factor bounded guarantees."""
    verification = VerificationResult(
        candidate_id="test-candidate",
        matched_required=["Python", "FastAPI"],
        matched_preferred=["Docker"],
        required_gaps=[],
        preferred_gaps=[],
        tech_coverage_pct=100.0,
    )
    profile = CandidateProfile(
        full_name="Jane Doe",
        years_experience=5.0,
        projects=[
            ProjectItem(name="API Service", description="FastAPI microservice", tech_tags=["Python"]),
            ProjectItem(name="Data ETL", description="Postgres pipeline", tech_tags=["SQL"]),
        ],
    )
    job_reqs = JobRequirements(
        role="Backend Engineer",
        experience_target_years=5,
        education_criteria="BS in Computer Science",
        mandatory_skills=["Python", "FastAPI"],
        preferred_skills=["Docker"],
        soft_skills=["Communication"],
        responsibilities=["Build APIs"],
        domain_tags=["Backend"],
    )

    result = calculate_final_score(
        verification=verification,
        candidate_profile=profile,
        job_requirements=job_reqs,
        jd_similarity=0.85,
    )

    # Invariants
    assert 0.0 <= result.final_score <= 100.0
    assert result.tier in [t[1] for t in TIER_THRESHOLDS]
    assert 0.0 <= result.raw_factors.technical_skills <= 100.0
    assert 0.0 <= result.raw_factors.experience_tenure <= 100.0
    assert 0.0 <= result.raw_factors.jd_similarity <= 100.0
    assert 0.0 <= result.raw_factors.project_relevance <= 100.0
    assert 0.0 <= result.raw_factors.education_certs <= 100.0

    # Mathematical consistency: sum of contributions equals final score
    computed_sum = round(sum(result.weighted_contributions.values()), 2)
    assert abs(computed_sum - result.final_score) < 0.01
