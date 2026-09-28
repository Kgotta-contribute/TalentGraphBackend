from app.schemas.evaluation import ScoringResult, ScoringFactors, ScoringWeights, TIER_THRESHOLDS, VerificationResult
from app.schemas.candidate import CandidateProfile
from app.schemas.job_requirements import JobRequirements

DEFAULT_WEIGHTS = ScoringWeights()  # technical=0.40, experience=0.25, jd_sim=0.20, projects=0.10, edu=0.05

def calculate_technical_score(matched_required: list[str], mandatory_skills: list[str], 
                               matched_preferred: list[str], preferred_skills: list[str]) -> float:
    """Pure deterministic calculation."""
    if not mandatory_skills:
        return 50.0
    required_ratio = len(matched_required) / len(mandatory_skills)
    preferred_ratio = (len(matched_preferred) / len(preferred_skills)) if preferred_skills else 0.5
    return min(100.0, (required_ratio * 0.8 + preferred_ratio * 0.2) * 100)

def calculate_experience_score(candidate_years: float, target_years: int) -> float:
    """Pure deterministic calculation."""
    if target_years <= 0:
        return 75.0
    ratio = candidate_years / target_years
    if ratio >= 1.5:
        return 100.0
    elif ratio >= 1.0:
        return 85.0 + (ratio - 1.0) * 30  
    elif ratio >= 0.7:
        return 60.0 + (ratio - 0.7) / 0.3 * 25
    else:
        return ratio / 0.7 * 60

def assign_tier(score: float) -> str:
    for threshold, tier_name in TIER_THRESHOLDS:
        if score >= threshold:
            return tier_name
    return "Not Recommended"

def calculate_final_score(
    verification: VerificationResult,
    candidate_profile: CandidateProfile,
    job_requirements: JobRequirements,
    jd_similarity: float,  # from pgvector cosine similarity (0-1), multiply by 100
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> ScoringResult:
    weights.validate_sum()
    
    raw_technical = calculate_technical_score(
        verification.matched_required, job_requirements.mandatory_skills,
        verification.matched_preferred, job_requirements.preferred_skills
    )
    raw_experience = calculate_experience_score(
        candidate_profile.years_experience, job_requirements.experience_target_years
    )
    raw_jd_similarity = jd_similarity * 100
    raw_projects = min(100.0, len(candidate_profile.projects) * 20)  # 5 projects = 100
    raw_education = 70.0  # baseline; +15 if degree matches criteria, +15 if certs
    
    raw = ScoringFactors(
        technical_skills=round(raw_technical, 2),
        experience_tenure=round(raw_experience, 2),
        jd_similarity=round(raw_jd_similarity, 2),
        project_relevance=round(raw_projects, 2),
        education_certs=round(raw_education, 2),
    )
    
    contributions = {
        "technical_skills": round(raw.technical_skills * weights.technical_skills, 2),
        "experience_tenure": round(raw.experience_tenure * weights.experience_tenure, 2),
        "jd_similarity": round(raw.jd_similarity * weights.jd_similarity, 2),
        "project_relevance": round(raw.project_relevance * weights.project_relevance, 2),
        "education_certs": round(raw.education_certs * weights.education_certs, 2),
    }
    final = round(sum(contributions.values()), 2)
    tier = assign_tier(final)
    
    return ScoringResult(
        candidate_id=candidate_profile.full_name,  # will be replaced with UUID
        mandate_id="",
        raw_factors=raw,
        weights=weights,
        weighted_contributions=contributions,
        final_score=final,
        tier=tier,
    )
