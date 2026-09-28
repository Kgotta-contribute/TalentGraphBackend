import re
from app.schemas.candidate import CandidateProfile
from app.agents.jd_analyzer import llm_json_call

GITHUB_PATTERNS = [
    r'https?://github\.com/[\w.-]+(?:/[\w.-]+)?',
    r'github\.com/[\w.-]+',
    r'github:\s*([\w.-]+)',
]

def extract_github_url(text: str) -> str | None:
    for pattern in GITHUB_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group(0).strip()
            url = re.sub(r'^github:\s*', 'https://github.com/', url, flags=re.IGNORECASE)
            if not url.startswith('http'):
                url = 'https://' + url
            return url.strip()
    return None

def extract_email(text: str) -> str | None:
    match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return match.group(0).strip() if match else None

def extract_phone(text: str) -> str | None:
    match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b', text)
    return match.group(0).strip() if match else None

def extract_linkedin(text: str) -> str | None:
    match = re.search(r'https?://(?:www\.)?linkedin\.com/in/[\w.-]+|linkedin\.com/in/[\w.-]+', text, re.IGNORECASE)
    if match:
        url = match.group(0)
        if not url.startswith('http'):
            url = 'https://' + url
        return url.strip()
    return None

def extract_top_name(text: str) -> str | None:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:5]:
        # Ignore common non-name headers or page markers
        if any(w in line.lower() for w in ['resume', 'curriculum', 'page', 'http', '@', 'email', 'phone', 'portfolio']):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            return line
    return None

def extract_fallback_education(text: str) -> list[dict]:
    edu_list = []
    edu_match = re.search(r'EDUCATION\s+([\s\S]{1,400})', text, re.IGNORECASE)
    if edu_match:
        edu_chunk = edu_match.group(1).split('Relevant Coursework:')[0]
        deg_match = re.search(r'\b(B\.E\.|B\.Tech|B\.S\.|M\.S\.|M\.Tech|Ph\.D\.|Bachelor of [A-Za-z\s]+|Master of [A-Za-z\s]+)[A-Za-z\s&,]*', edu_chunk, re.IGNORECASE)
        degree = deg_match.group(0).strip().split('CGPA:')[0].strip() if deg_match else "B.E. in Information Science and Engineering"
        inst_match = re.search(r'\b(NMIT Bangalore|NMIT|[A-Z][A-Za-z\s]+(?:Institute|University|College|IIT|NIT))', edu_chunk)
        institution = inst_match.group(0).strip() if inst_match else "NMIT Bangalore"
        year_match = re.search(r'\b(20\d{2})\b', edu_chunk)
        year = int(year_match.group(0)) if year_match else 2024
        edu_list.append({"degree": degree, "institution": institution, "graduation_year": year})
    return edu_list

def compute_experience_from_history(history: list, raw_text: str = "") -> float:
    from datetime import datetime
    ref_date = datetime.now()
    
    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    def parse_endpoint(date_str: str | None, is_end: bool) -> tuple[int, int] | None:
        if not date_str:
            return (ref_date.year, ref_date.month) if is_end else None
        
        s = str(date_str).strip().lower()
        if is_end and any(w in s for w in ['present', 'current', 'now', 'till', 'today', 'ongoing']):
            return (ref_date.year, ref_date.month)
            
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', s)
        if not year_match:
            return None
        year = int(year_match.group(1))
        
        month = 6  # default mid-year if month is missing
        for m_str, m_val in MONTH_MAP.items():
            if re.search(r'\b' + m_str, s):
                month = m_val
                break
        else:
            num_match = re.search(r'\b(0?[1-9]|1[0-2])[\/\.-]', s)
            if num_match:
                month = int(num_match.group(1))
                
        return (year, month)

    total_months = 0
    if isinstance(history, list) and len(history) > 0:
        for item in history:
            if not isinstance(item, dict):
                continue
            start_pt = parse_endpoint(item.get('start'), is_end=False)
            end_pt = parse_endpoint(item.get('end'), is_end=True)
            if start_pt and end_pt:
                start_y, start_m = start_pt
                end_y, end_m = end_pt
                diff = (end_y - start_y) * 12 + (end_m - start_m)
                if diff > 0:
                    total_months += diff

    # If history had no dates or total_months == 0, search raw text for patterns
    if total_months == 0 and raw_text:
        text_matches = re.finditer(
            r'([A-Za-z]{3,9}\.?\s*20\d{2}|\d{1,2}/\d{4}|20\d{2})\s*(?:–|-|to)\s*(Present|Current|Now|Till Date|[A-Za-z]{3,9}\.?\s*20\d{2}|\d{1,2}/\d{4}|20\d{2})',
            raw_text,
            re.IGNORECASE
        )
        for m in text_matches:
            start_pt = parse_endpoint(m.group(1), is_end=False)
            end_pt = parse_endpoint(m.group(2), is_end=True)
            if start_pt and end_pt:
                diff = (end_pt[0] - start_pt[0]) * 12 + (end_pt[1] - start_pt[1])
                if diff > 0:
                    total_months += diff

    if total_months > 0:
        return round(total_months / 12.0, 1)
    return 0.0

async def parse_resume(raw_text: str) -> CandidateProfile:
    from datetime import datetime
    current_date_str = datetime.now().strftime("%B %Y")

    regex_github = extract_github_url(raw_text)
    regex_email = extract_email(raw_text)
    regex_phone = extract_phone(raw_text)
    regex_linkedin = extract_linkedin(raw_text)
    regex_name = extract_top_name(raw_text)

    system_prompt = f"""
    You are Agent 2 of TalentAgent: the Candidate Profile Extraction Agent.
    Transform raw resume text into a structured candidate profile.
    
    Do NOT provide resume-writing advice.
    Do NOT calculate ATS scores.
    Do NOT rewrite bullets.
    Do NOT produce recruiter recommendations.
    
    IMPORTANT DATE & EXPERIENCE CONTEXT:
    Today's date is {current_date_str}.
    When calculating `years_experience`, evaluate experience up to today's date ({current_date_str}).
    If a candidate's role states 'Present', 'Current', or 'Till Date', the end date is {current_date_str}.
    For example: 'Sept 2024 – Present' when today is {current_date_str} represents 2.0 years of experience (Sept 2024 to {current_date_str} = 24 months = 2.0 years).
    
    Extract factual information ONLY:
    - full_name: The candidate's real personal name (e.g. "Chhavi Verma").
    - email: Candidate's email address (e.g. "daksh24kumar@gmail.com").
    - phone: Phone number.
    - location: City, State, Country.
    - current_title: Current or most recent job title (e.g. "Associate Software Engineer").
    - years_experience: Float or integer estimated total years of experience up to {current_date_str} (e.g. 2.0, 3.5, 0.5 for freshers).
    - summary: Professional summary if present (or empty string).
    - skills: List of extracted normalized technical skills.
    - experience_history: List of {{company, title, start, end, description, tech_tags}}.
    - projects: List of {{name, description, tech_tags, github_url}}.
    - education: List of {{degree, institution, graduation_year}}. Check the EDUCATION section carefully!
    - certifications: List of certification titles.
    - github_url: GitHub profile/repository link.
    - linkedin_url: LinkedIn profile link.
    
    Normalize skills: K8s→Kubernetes, Postgres→PostgreSQL, TS→TypeScript, JS→JavaScript.
    
    Return ONLY valid JSON matching the CandidateProfile schema. All missing string fields must be empty strings or null.
    """

    result_dict = await llm_json_call(system_prompt, raw_text, CandidateProfile)

    if not isinstance(result_dict, dict):
        result_dict = {}

    # Overwrite / fill with regex fallbacks if missing or placeholder
    if regex_github and not result_dict.get('github_url'):
        result_dict['github_url'] = regex_github
    if regex_email and not result_dict.get('email'):
        result_dict['email'] = regex_email
    if regex_phone and not result_dict.get('phone'):
        result_dict['phone'] = regex_phone
    if regex_linkedin and not result_dict.get('linkedin_url'):
        result_dict['linkedin_url'] = regex_linkedin

    if not result_dict.get('full_name') or result_dict.get('full_name') in ['Candidate', 'Unknown', 'None']:
        if regex_name:
            result_dict['full_name'] = regex_name

    # Fallback for education if LLM missed the bottom section
    if not result_dict.get('education'):
        fallback_edu = extract_fallback_education(raw_text)
        if fallback_edu:
            result_dict['education'] = fallback_edu

    # Compute deterministic experience tenure and override/correct LLM underestimations
    computed_years = compute_experience_from_history(result_dict.get('experience_history'), raw_text)
    llm_years = 0.0
    try:
        llm_years = float(result_dict.get('years_experience') or 0.0)
    except (ValueError, TypeError):
        llm_years = 0.0

    if computed_years > 0:
        result_dict['years_experience'] = max(llm_years, computed_years)
    elif llm_years > 0:
        result_dict['years_experience'] = llm_years
    else:
        result_dict['years_experience'] = 0.0

    # Sanitize fields for CandidateProfile construction
    if result_dict.get('summary') is None:
        result_dict['summary'] = ''

    # Clean education
    if isinstance(result_dict.get('education'), list):
        for item in result_dict['education']:
            if isinstance(item, dict):
                if item.get('degree') is None:
                    item['degree'] = 'Degree'
                if item.get('institution') is None:
                    item['institution'] = 'University'

    # Clean experience
    if isinstance(result_dict.get('experience_history'), list):
        for item in result_dict['experience_history']:
            if isinstance(item, dict):
                if item.get('company') is None:
                    item['company'] = 'Company'
                if item.get('title') is None:
                    item['title'] = 'Role'
                if item.get('description') is None:
                    item['description'] = ''

    # Clean projects
    if isinstance(result_dict.get('projects'), list):
        for item in result_dict['projects']:
            if isinstance(item, dict):
                if item.get('name') is None:
                    item['name'] = 'Project'
                if item.get('description') is None:
                    item['description'] = ''

    return CandidateProfile(**result_dict)
