import asyncio
import json
import logging
from app.schemas.job_requirements import JobRequirements
from app.core.llm import invoke_json, invoke_structured_output

logger = logging.getLogger("talent_agent.langchain_llm")

async def llm_json_call(system_prompt: str, user_content: str, response_schema: type = None) -> dict:
    """
    Unified LangChain JSON caller with multi-provider support (Groq/OpenAI/Anthropic),
    automatic model fallback, and rate limiting.
    """
    return await invoke_json(system_prompt=system_prompt, user_content=user_content)


async def analyze_jd(raw_jd: str) -> JobRequirements:
    system_prompt = """
    You are Agent 1 of TalentAgent: the Job Description Analyzer.

    IMPORTANT — INPUT NOISE HANDLING:
    The input may contain webpage or social-media noise such as:
    - LinkedIn UI buttons: "Apply", "Save", "Show match details", "Tailor my resume"
    - Platform metadata: "Reposted 2 days ago", "Over 100 people clicked apply",
      "Promoted by hirer", "Responses managed off LinkedIn"
    - Tracking URLs, markdown links, navigation labels, filter tags
    IGNORE all of this noise. Evaluate ONLY the substantive employment-related content.

    STEP 1 — VALIDITY CHECK (mandatory):
    After ignoring noise, determine whether the remaining content is a genuine job description
    intended to recruit a candidate for an employment role. A valid JD describes responsibilities,
    required skills, qualifications, or employment terms.

    If the content is NOT a job description (e.g. song lyrics, poem, news article, recipe,
    random text, abusive content), return ONLY:
    {"is_valid_jd": false, "rejection_reason": "<brief reason>"}

    STEP 2 — EXTRACTION (only if valid):
    Extract the following fields from the job description content:
    - role: job title (e.g. "AI-Driven Full Stack Engineer", "Senior Software Engineer")
    - company: company or organization name (e.g. "Infosys", "Tekion", "Amazon"; check headers, markdown links like [Infosys](https://...linkedin.com/company/infosys...), or mention in text; null if unknown)
    - experience_target_years: integer years (e.g. 3, 5; if not explicitly stated, infer from seniority e.g. 3 for mid-level, 5 for senior; NEVER return null)
    - education_criteria: degree requirement string (if not stated, use "Bachelor's degree in Computer Science or equivalent practical experience")
    - mandatory_skills: skills explicitly required for successful performance
    - preferred_skills: skills described as preferred, bonus, nice-to-have, or plus
    - soft_skills: interpersonal/soft skills
    - responsibilities: main job responsibilities
    - domain_tags: domain labels (backend, frontend, AI/ML, cloud, databases, etc.)

    Normalization rules:
    - K8s → Kubernetes  |  Postgres → PostgreSQL  |  TS → TypeScript  |  JS → JavaScript

    Do NOT fabricate requirements. Do NOT include noise text in any field.
    For valid JDs include "is_valid_jd": true plus all required fields.
    Return ONLY valid JSON.
    """
    result_dict = await llm_json_call(system_prompt, raw_jd, JobRequirements)

    # Layer 3 semantic guard: reject if LLM says input is not a JD
    if not result_dict.get("is_valid_jd", True):
        reason = result_dict.get("rejection_reason", "Input does not appear to be a job description.")
        raise ValueError(f"Semantic validation failed: {reason}")

    # Fallback company detection via regex if LLM missed it
    if not result_dict.get("company"):
        import re
        m_link = re.search(r'\[([A-Za-z0-9&.,\s-]{2,40})\]\(https?://(?:www\.)?linkedin\.com/company/', raw_jd)
        if m_link:
            result_dict["company"] = m_link.group(1).strip()
        else:
            m_comp = re.search(r'(?:Company|Employer|Organization)\s*[-:]\s*([A-Za-z0-9&.,\s-]{2,40})', raw_jd, re.IGNORECASE)
            if m_comp:
                result_dict["company"] = m_comp.group(1).strip().split('\n')[0].strip()

    # Sanitize experience_target_years
    exp = result_dict.get("experience_target_years")
    if exp is None:
        result_dict["experience_target_years"] = 3
    elif isinstance(exp, str):
        import re
        nums = re.findall(r'\d+', exp)
        result_dict["experience_target_years"] = int(nums[0]) if nums else 3
    elif isinstance(exp, (float, int)):
        result_dict["experience_target_years"] = int(exp)
    else:
        result_dict["experience_target_years"] = 3

    # Sanitize education_criteria
    if not result_dict.get("education_criteria"):
        result_dict["education_criteria"] = "Bachelor's degree in Computer Science, Engineering, or equivalent practical experience"

    # Sanitize role
    if not result_dict.get("role"):
        result_dict["role"] = "Software Engineer"

    # Ensure list types are lists
    for key in ["mandatory_skills", "preferred_skills", "soft_skills", "responsibilities", "domain_tags"]:
        if not isinstance(result_dict.get(key), list):
            result_dict[key] = []

    return JobRequirements(**result_dict)
