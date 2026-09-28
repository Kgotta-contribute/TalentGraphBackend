from app.schemas.report import RecruitmentDossier
from app.agents.jd_analyzer import llm_json_call
import json


async def generate_report(
    candidate_id: str,
    candidate_profile: dict,
    verification_result: dict,
    scoring_result: dict,
    github_analysis: dict | None = None,
) -> RecruitmentDossier:
    """
    Agent 5: Recruitment Intelligence Dossier Generator.
    
    Synthesizes verified evidence into a recruiter-facing evaluation.
    DOES NOT modify the deterministic score from Agent 4.
    """
    system_prompt = """
    You are Agent 5 of TalentAgent: Recruitment Intelligence Dossier Agent.
    Synthesize verified evidence into a recruiter-facing evaluation.
    
    You are NOT allowed to invent candidate facts.
    
    Rules:
    - The deterministic score from Agent 4 is AUTHORITATIVE. Do NOT change it.
    - Do NOT invent missing skills.
    - Do NOT describe GitHub as proof of employment.
    - Distinguish resume evidence from GitHub evidence.
    - Treat missing GitHub as unknown, NOT negative evidence.
    - Do NOT make discriminatory or protected-attribute inferences.
    - hiring_confidence must be a float between 0.0 and 1.0
    
    For interview questions:
    - Every question must trace to a requirement, gap, project, or candidate claim.
    - Generate at least 5 interview questions.
    - At least 1 question tests a required skill gap.
    - At least 1 question tests technical depth.
    - At least 1 question tests project-level engineering judgment.
    - Each question needs: focus_area (string), question (string), rationale (string), keywords (list of strings).
    
    Output JSON must match exactly:
    {
      "candidate_id": "<string>",
      "executive_summary": "<string>",
      "key_strengths": ["<string>"],
      "identified_skill_gaps": ["<string>"],
      "risk_factors": ["<string>"],
      "ramp_up_considerations": ["<string>"],
      "final_verdict": "<string>",
      "hiring_confidence": <float 0.0-1.0>,
      "relevant_experience": [{"title": "<string>", "company": "<string>", "impact": "<string>"}],
      "relevant_projects": [{"name": "<string>", "relevance": "<string>"}],
      "github_summary": "<string or null>",
      "interview_questions": [
        {
          "focus_area": "<string>",
          "question": "<string>",
          "rationale": "<string>",
          "keywords": ["<string>"]
        }
      ]
    }
    
    Return ONLY valid JSON.
    """

    user_content = json.dumps({
        "candidate_id": candidate_id,
        "candidate_profile": candidate_profile,
        "verification_result": verification_result,
        "scoring_result": scoring_result,
        "github_analysis": github_analysis,
    })

    result_dict = await llm_json_call(system_prompt, user_content, RecruitmentDossier)
    return RecruitmentDossier(**result_dict)
