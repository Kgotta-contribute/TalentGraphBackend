from app.schemas.evaluation import VerificationResult
from app.schemas.job_requirements import JobRequirements
from app.schemas.candidate import CandidateProfile
from app.agents.jd_analyzer import llm_json_call
import json

async def verify_requirements(
    job_requirements: JobRequirements,
    candidate_profile: CandidateProfile,
    github_analysis: dict | None = None,
    relevant_chunks: list[dict] = [],
    candidate_id: str = ""
) -> VerificationResult:
    system_prompt = """
    You are Agent 3 of TalentAgent: Requirement Verification Agent.
    
    Verify whether a candidate's evidence supports the job mandate requirements.
    You are NOT ranking candidates and NOT making hiring decisions.
    
    Rules:
    - Never invent candidate experience
    - Distinguish REQUIRED from PREFERRED requirements
    - A missing GitHub URL is NOT evidence of a skill gap
    - GitHub evidence strengthens verification but must not invent experience
    - Forked repositories with little/no original activity are NOT strong evidence
    - Use PARTIAL or UNKNOWN when evidence is ambiguous, not MATCHED
    - Return evidence for every important match or gap
    - Do NOT produce a numeric candidate ranking
    
    Output JSON format:
    {
      "matched_required": ["<string>"],
      "matched_preferred": ["<string>"],
      "required_gaps": ["<string>"],
      "preferred_gaps": ["<string>"],
      "partial_matches": ["<string>"],
      "evidence": [
        {
          "requirement": "<string>",
          "status": "MATCHED",  // MATCHED | PARTIAL | MISSING | UNKNOWN
          "source": "RESUME",    // RESUME | PROJECT | GITHUB | EDUCATION
          "evidence": "<string>",
          "confidence": 0.85
        }
      ],
      "tech_coverage_pct": 80.0
    }
    
    Return strict JSON matching this structure.
    """
    
    user_content = json.dumps({
        "job_requirements": job_requirements.model_dump(),
        "candidate_profile": candidate_profile.model_dump(),
        "github_analysis": github_analysis,
        "relevant_chunks": relevant_chunks
    })
    
    result_dict = await llm_json_call(system_prompt, user_content, VerificationResult)
    result_dict["candidate_id"] = candidate_id
    
    # Deterministic fallback calculation for tech_coverage_pct if needed
    mandatory = job_requirements.mandatory_skills or []
    matched = result_dict.get("matched_required", [])
    if mandatory:
        result_dict["tech_coverage_pct"] = round((len(matched) / len(mandatory)) * 100.0, 1)
    elif not result_dict.get("tech_coverage_pct"):
        result_dict["tech_coverage_pct"] = 75.0
        
    return VerificationResult(**result_dict)

