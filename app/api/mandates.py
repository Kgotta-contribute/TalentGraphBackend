import logging
from datetime import datetime
import uuid as uuid_mod
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user
from app.models.mandate import Mandate
from app.models.candidate import Candidate
from app.models.evaluation import CandidateEvaluation
from app.models.analysis import AnalysisRun
from app.agents.jd_analyzer import analyze_jd
from app.ranking.scoring import calculate_final_score, DEFAULT_WEIGHTS
from app.schemas.evaluation import ScoringWeights
from app.schemas.job_requirements import JobRequirements
from app.schemas.candidate import CandidateProfile

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request bodies ───────────────────────────────────────────────────────────

class CreateMandateBody(BaseModel):
    title: str
    company: str


class SaveJDBody(BaseModel):
    jd: str
    title: str | None = None
    company: str | None = None


class UpdateWeightsBody(BaseModel):
    technical_skills: float
    experience_tenure: float
    jd_similarity: float
    project_relevance: float
    education_certs: float


# ─── Helpers ──────────────────────────────────────────────────────────────────

import re as _re

def mandate_to_dict(m: Mandate) -> dict:
    return {
        "id": str(m.id),
        "recruiter_id": m.recruiter_id,
        "title": m.title,
        "company": m.company,
        "raw_jd": m.raw_jd,
        "status": m.status,
        "job_requirements": m.job_requirements,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }


# ─── JD Validator — 2-Stage Pipeline (Layer 2) ───────────────────────────────

_ABUSIVE = [
    _re.compile(r'\bf+u+c+k+\b', _re.IGNORECASE),
    _re.compile(r'\bs+h+i+t+\b', _re.IGNORECASE),
    _re.compile(r'\bb+i+t+c+h+\b', _re.IGNORECASE),
    _re.compile(r'\ba+s+s+h+o+l+e+\b', _re.IGNORECASE),
    _re.compile(r'\bc+u+n+t+\b', _re.IGNORECASE),
    _re.compile(r'\bw+h+o+r+e+\b', _re.IGNORECASE),
]

_SONG_MARKERS = _re.compile(
    r'\[(chorus|verse|bridge|outro|intro|refrain|hook)\]|\brefrain:\b|\(repeat\)|\bla la la\b',
    _re.IGNORECASE,
)

_NON_JD_SIGNALS = [
    _re.compile(r'\b(breaking news|live updates|latest news|advertisement|ePaper)\b', _re.IGNORECASE),
    _re.compile(r'copyright ©.*all rights reserved', _re.IGNORECASE),
    _re.compile(r'skip to (content|main)', _re.IGNORECASE),
    _re.compile(r"\b(today's paper|journalism of courage|express premium)\b", _re.IGNORECASE),
    _re.compile(r'medal tally|kabaddi|asian games', _re.IGNORECASE),
    _re.compile(
        r'\b(sports|cricket|bollywood|politics|opinion|trending)\b.{0,60}'
        r'\b(sports|cricket|bollywood|politics|opinion|trending)\b.{0,60}'
        r'\b(sports|cricket|bollywood|politics|opinion|trending)\b',
        _re.IGNORECASE | _re.DOTALL,
    ),
]

# Stage 1: noise line detection
_NOISE_EXACT = {
    'apply', 'save', 'premium', 'show all', 'on-site', 'full-time', 'part-time',
    'hybrid', 'remote', 'contract', 'temporary',
    'people you can reach out to', 'responses managed off linkedin',
    'promoted by hirer', 'tailor my resume', 'create cover letter',
    'help me stand out', 'show match details',
    'determine your fit and how to stand out', 'about the company',
    'see all', 'easy apply', 'apply now', 'apply on company website',
    'be an early applicant', 'actively recruiting',
}
_NOISE_PAT = [
    _re.compile(r'^reposted \d+ (day|week|month)s? ago$', _re.IGNORECASE),
    _re.compile(r'^over \d+ people (clicked|applied|viewed)', _re.IGNORECASE),
    _re.compile(r'^https?://', _re.IGNORECASE),
    _re.compile(r'^\[.+\]\(https?://.+\)$'),
    _re.compile(r'^\d+ (applicants?|connections?|followers?)$', _re.IGNORECASE),
]


def _is_noise_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t.lower() in _NOISE_EXACT:
        return True
    return any(p.match(t) for p in _NOISE_PAT)


# Stage 2: JD evidence signals
_JD_EVIDENCE = [
    'about the job', 'role description', 'job description',
    'key responsibilities', 'responsibilities',
    'what you will do', "what you'll do", 'your responsibilities',
    'required qualifications', 'minimum qualifications', 'basic qualifications',
    'preferred qualifications', 'qualifications', 'requirements',
    'technical skills', 'skills required', 'required skills',
    "what we're looking for", 'what we are looking for',
    'desired competencies', 'success measures',
    'about the role', 'about this role',
    "bachelor's degree", "master's degree",
    'years of experience', 'years experience', '+ years',
    'equivalent experience', 'phd',
    'equal opportunity employer', 'we are looking for', 'we are hiring',
    'in this role', 'you will be responsible',
    'software engineer', 'data engineer', 'product manager', 'engineering manager',
    'full stack', 'fullstack', 'ml engineer', 'senior engineer', 'staff engineer',
    'principal engineer', 'product engineer', 'solutions architect', 'tech lead',
]


def _validate_jd_heuristic(text: str) -> str | None:
    """Return an error reason string if text is not a valid job description, else None."""
    stripped = text.strip()
    if len(stripped) < 120:
        return None

    # Safety checks on raw text
    for pattern in _ABUSIVE:
        if pattern.search(stripped):
            return "Abusive or offensive content detected."

    if _SONG_MARKERS.search(stripped):
        return "Text appears to contain song lyrics or poetry markers."

    for pattern in _NON_JD_SIGNALS:
        if pattern.search(stripped):
            return "Text appears to be a news article or website content, not a job description."

    url_count = len(_re.findall(r'https?://', stripped))
    if url_count > 30:
        return f"Too many URLs detected ({url_count}). Please paste plain-text job description only."

    # Stage 1: strip noise lines
    cleaned = '\n'.join(l for l in stripped.split('\n') if not _is_noise_line(l))
    cleaned_lower = cleaned.lower()

    if len(cleaned.split()) < 30:
        return "Text is too sparse after removing navigation/UI noise. Please paste the actual job description body."

    # Stage 2: JD evidence detection
    matched = [s for s in _JD_EVIDENCE if s in cleaned_lower]
    if len(matched) >= 2:
        return None  # valid

    found = f'Found only: "{matched[0]}". ' if matched else ''
    return (
        f"{found}Non-job content detected: text lacks job description sections "
        "such as \"Responsibilities\", \"Qualifications\", or \"About the job\"."
    )


def eval_to_dict(e: CandidateEvaluation, candidate: Candidate | None = None) -> dict:
    d = {
        "id": str(e.id),
        "mandate_id": str(e.mandate_id),
        "candidate_id": str(e.candidate_id),
        "verification_result": e.verification_result,
        "tech_coverage_pct": e.tech_coverage_pct,
        "github_analysis": e.github_analysis,
        "github_verified": e.github_verified,
        "score_technical": e.score_technical,
        "score_experience": e.score_experience,
        "score_jd_similarity": e.score_jd_similarity,
        "score_projects": e.score_projects,
        "score_education": e.score_education,
        "final_score": e.final_score,
        "tier": e.tier,
        "scoring_weights": e.scoring_weights,
        "report": e.report,
        "status": e.status,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }
    if candidate:
        d["candidate"] = {
            "id": str(candidate.id),
            "full_name": candidate.full_name,
            "email": candidate.email,
            "github_url": candidate.github_url,
            "current_title": candidate.current_title,
            "years_experience": candidate.years_experience,
            "profile": candidate.profile,
            "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
        }
    return d


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("")
async def create_mandate(
    body: CreateMandateBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    mandate = Mandate(
        recruiter_id=current_user["id"],
        title=body.title,
        company=body.company,
    )
    db.add(mandate)
    await db.commit()
    await db.refresh(mandate)
    return mandate_to_dict(mandate)


@router.get("")
async def list_mandates(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(
        select(Mandate).where(Mandate.recruiter_id == current_user["id"])
        .order_by(Mandate.created_at.desc())
    )
    mandates = result.scalars().all()
    return [mandate_to_dict(m) for m in mandates]


@router.get("/{mandate_id}")
async def get_mandate(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(select(Mandate).where(Mandate.id == uuid_mod.UUID(mandate_id)))
    mandate = result.scalar_one_or_none()
    if not mandate:
        raise HTTPException(status_code=404, detail="Mandate not found")
    return mandate_to_dict(mandate)


@router.put("/{mandate_id}/job-description")
async def save_jd(
    mandate_id: str,
    body: SaveJDBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(select(Mandate).where(Mandate.id == uuid_mod.UUID(mandate_id)))
    mandate = result.scalar_one_or_none()
    if not mandate:
        raise HTTPException(status_code=404, detail="Mandate not found")
    mandate.raw_jd = body.jd
    if body.title:
        mandate.title = body.title
    if body.company is not None:
        mandate.company = body.company
    await db.commit()
    await db.refresh(mandate)
    return mandate_to_dict(mandate)


@router.post("/{mandate_id}/analyze-jd")
async def analyze_jd_route(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(select(Mandate).where(Mandate.id == uuid_mod.UUID(mandate_id)))
    mandate = result.scalar_one_or_none()
    if not mandate:
        raise HTTPException(status_code=404, detail="Mandate not found")
    if not mandate.raw_jd:
        raise HTTPException(status_code=400, detail="No JD text saved yet")

    # ── Layer 2: Backend heuristic JD guardrail ───────────────────────────────
    rejection = _validate_jd_heuristic(mandate.raw_jd)
    if rejection:
        raise HTTPException(status_code=422, detail=f"Invalid Job Description: {rejection}")

    # Run Agent 1
    mandate.status = "analyzing"
    await db.commit()
    try:
        requirements = await analyze_jd(mandate.raw_jd)
        mandate.job_requirements = requirements.model_dump()
        if requirements.role:
            mandate.title = requirements.role

        if requirements.company:
            mandate.company = requirements.company
        else:
            # Extract company from JD if present via regex
            comp_match = _re.search(
                r'(?:Company|Employer|Organization)\s*[-:]\s*([A-Za-z0-9&.,\s-]{2,40})',
                mandate.raw_jd,
                _re.IGNORECASE,
            )
            if comp_match:
                cand_comp = comp_match.group(1).strip().split('\n')[0].strip()
                if cand_comp:
                    mandate.company = cand_comp
            else:
                link_match = _re.search(
                    r'\[([A-Za-z0-9&.,\s-]{2,40})\]\(https?://(?:www\.)?linkedin\.com/company/',
                    mandate.raw_jd,
                )
                if link_match:
                    mandate.company = link_match.group(1).strip()
                else:
                    about_match = _re.search(
                        r'About\s+([A-Z][A-Za-z0-9&.,\s-]{1,30})',
                        mandate.raw_jd,
                    )
                    if about_match:
                        val = about_match.group(1).strip()
                        if val.lower() not in ['the role', 'the job', 'the position', 'our team', 'this role', 'us', 'you', 'rippling', 'amazon']:
                            mandate.company = val
                        elif val.lower() in ['rippling', 'amazon']:
                            mandate.company = val.capitalize()

        mandate.status = "active"
        await db.commit()
        await db.refresh(mandate)
        return mandate.job_requirements
    except Exception as e:
        mandate.status = "draft"
        await db.commit()
        raise HTTPException(status_code=500, detail=f"JD analysis failed: {str(e)}")


async def ensure_mandate_evaluations(db: AsyncSession, mandate_id: uuid_mod.UUID) -> list[CandidateEvaluation]:
    """
    Ensures that all candidates in the global pool have an evaluation row
    for this mandate. Automatically syncs any newly uploaded candidate.
    """
    result = await db.execute(
        select(CandidateEvaluation)
        .where(CandidateEvaluation.mandate_id == mandate_id)
    )
    evals = list(result.scalars().all())

    all_cands_res = await db.execute(select(Candidate))
    all_cands = all_cands_res.scalars().all()
    existing_cand_ids = {ev.candidate_id for ev in evals}

    new_added = False
    for c in all_cands:
        if c.id not in existing_cand_ids:
            new_ev = CandidateEvaluation(
                id=uuid_mod.uuid4(),
                mandate_id=mandate_id,
                candidate_id=c.id,
                status="parsed",
            )
            db.add(new_ev)
            evals.append(new_ev)
            new_added = True

    if new_added:
        await db.commit()
        result = await db.execute(
            select(CandidateEvaluation)
            .where(CandidateEvaluation.mandate_id == mandate_id)
        )
        evals = list(result.scalars().all())

    return evals


@router.get("/{mandate_id}/candidates")
async def list_candidates(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    mid = uuid_mod.UUID(mandate_id)
    evals = await ensure_mandate_evaluations(db, mid)

    enriched = []
    for ev in evals:
        cand_res = await db.execute(select(Candidate).where(Candidate.id == ev.candidate_id))
        candidate = cand_res.scalar_one_or_none()
        enriched.append(eval_to_dict(ev, candidate))
    return enriched


@router.post("/{mandate_id}/verify")
async def verify_candidates(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Agent 3: Run requirement verification for all candidates in the mandate."""
    m_result = await db.execute(select(Mandate).where(Mandate.id == uuid_mod.UUID(mandate_id)))
    mandate = m_result.scalar_one_or_none()
    if not mandate or not mandate.job_requirements:
        raise HTTPException(status_code=400, detail="Mandate has no job requirements. Run JD analysis first.")

    job_req = JobRequirements(**mandate.job_requirements)
    evals = await ensure_mandate_evaluations(db, uuid_mod.UUID(mandate_id))

    verified_count = 0
    from app.agents.requirement_verifier import verify_requirements

    for ev in evals:
        # Skip if already verified to optimize speed and API quota
        if ev.verification_result and ev.tech_coverage_pct is not None:
            continue

        cand_res = await db.execute(select(Candidate).where(Candidate.id == ev.candidate_id))
        candidate = cand_res.scalar_one_or_none()
        if not candidate or not candidate.profile:
            continue

        cand_profile = CandidateProfile(**candidate.profile)
        try:
            verification = await verify_requirements(
                job_requirements=job_req,
                candidate_profile=cand_profile,
                github_analysis=ev.github_analysis,
            )
            ev.verification_result = verification.model_dump()
            ev.tech_coverage_pct = verification.tech_coverage_pct
        except Exception as e:
            logger.warning(f"Error in Agent 3 verification for candidate {candidate.full_name}: {e}", exc_info=True)
            # Deterministic fallback matching
            mandatory = job_req.mandatory_skills or []
            cand_skills = [s.lower() for s in (cand_profile.skills or [])]
            matched = [s for s in mandatory if any(c_sk in s.lower() or s.lower() in c_sk for c_sk in cand_skills)]
            coverage = round((len(matched) / len(mandatory) * 100.0) if mandatory else 75.0, 1)
            ev.verification_result = {
                "candidate_id": str(candidate.id),
                "matched_required": matched,
                "matched_preferred": [],
                "required_gaps": [s for s in mandatory if s not in matched],
                "preferred_gaps": [],
                "partial_matches": [],
                "evidence": [],
                "tech_coverage_pct": coverage
            }
            ev.tech_coverage_pct = coverage

        ev.status = "verified"
        await db.commit()
        await db.refresh(ev)
        verified_count += 1

    return {"message": f"Verified {verified_count} candidates using Agent 3"}


@router.post("/{mandate_id}/rank")
async def rank_candidates(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Agent 4: Deterministic ranking for all candidates in a mandate."""
    m_result = await db.execute(select(Mandate).where(Mandate.id == uuid_mod.UUID(mandate_id)))
    mandate = m_result.scalar_one_or_none()
    if not mandate or not mandate.job_requirements:
        raise HTTPException(status_code=400, detail="Mandate has no job requirements. Run JD analysis first.")

    job_req = JobRequirements(**mandate.job_requirements)
    evals = await ensure_mandate_evaluations(db, uuid_mod.UUID(mandate_id))

    ranked = []
    for ev in evals:
        if not ev.verification_result or not ev.candidate_id:
            continue
        cand_res = await db.execute(select(Candidate).where(Candidate.id == ev.candidate_id))
        candidate = cand_res.scalar_one_or_none()
        if not candidate or not candidate.profile:
            continue

        from app.schemas.evaluation import VerificationResult
        verification = VerificationResult(**ev.verification_result)
        profile = CandidateProfile(**candidate.profile)

        # Use stored weights or defaults
        weights = ScoringWeights(**(ev.scoring_weights or {})) if ev.scoring_weights else DEFAULT_WEIGHTS

        scoring = calculate_final_score(
            verification=verification,
            candidate_profile=profile,
            job_requirements=job_req,
            jd_similarity=ev.tech_coverage_pct / 100.0 if ev.tech_coverage_pct else 0.5,
            weights=weights,
        )

        ev.score_technical = scoring.raw_factors.technical_skills
        ev.score_experience = scoring.raw_factors.experience_tenure
        ev.score_jd_similarity = scoring.raw_factors.jd_similarity
        ev.score_projects = scoring.raw_factors.project_relevance
        ev.score_education = scoring.raw_factors.education_certs
        ev.final_score = scoring.final_score
        ev.tier = scoring.tier
        ev.status = "ranked"

        await db.commit()
        await db.refresh(ev)
        ranked.append(eval_to_dict(ev, candidate))

    ranked.sort(key=lambda x: x.get("final_score") or 0, reverse=True)
    return ranked


@router.get("/{mandate_id}/ranking")
async def get_ranking(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    mid = uuid_mod.UUID(mandate_id)
    evals = await ensure_mandate_evaluations(db, mid)

    enriched = []
    for ev in evals:
        cand_res = await db.execute(select(Candidate).where(Candidate.id == ev.candidate_id))
        candidate = cand_res.scalar_one_or_none()
        enriched.append(eval_to_dict(ev, candidate))

    # Sort by score descending
    enriched.sort(key=lambda x: x.get("final_score") or 0, reverse=True)
    return enriched


@router.put("/{mandate_id}/scoring-profile")
async def update_scoring(
    mandate_id: str,
    body: UpdateWeightsBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Validate weights sum to 1.0
    total = sum([
        body.technical_skills, body.experience_tenure, body.jd_similarity,
        body.project_relevance, body.education_certs
    ])
    if abs(total - 1.0) > 0.001:
        raise HTTPException(status_code=400, detail=f"Weights must sum to 1.0, got {total:.3f}")

    # Apply to all evaluations in this mandate
    result = await db.execute(
        select(CandidateEvaluation)
        .where(CandidateEvaluation.mandate_id == uuid_mod.UUID(mandate_id))
    )
    evals = result.scalars().all()
    weights_dict = body.model_dump()
    for ev in evals:
        ev.scoring_weights = weights_dict
    await db.commit()
    return {"message": f"Scoring weights updated for {len(evals)} candidates"}


@router.post("/{mandate_id}/analysis-runs")
async def start_run(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    m_result = await db.execute(select(Mandate).where(Mandate.id == uuid_mod.UUID(mandate_id)))
    mandate = m_result.scalar_one_or_none()
    if not mandate:
        raise HTTPException(status_code=404, detail="Mandate not found")

    run = AnalysisRun(
        mandate_id=uuid_mod.UUID(mandate_id),
        run_type="full",
        status="running",
        started_at=datetime.utcnow(),
        agent1_status="completed",
        agent2_status="completed",
        agent3_status="running",
        agent4_status="pending",
        agent5_status="pending",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # Execute verification and ranking pipeline
    try:
        if mandate.job_requirements:
            await verify_candidates(mandate_id=mandate_id, db=db, current_user=current_user)
            run.agent3_status = "completed"
            run.agent4_status = "running"
            await db.commit()

            await rank_candidates(mandate_id=mandate_id, db=db, current_user=current_user)
            run.agent4_status = "completed"
            run.agent5_status = "completed"
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            await db.commit()
        else:
            run.status = "failed"
            run.errors = ["Mandate has no job requirements. Please analyze the JD first."]
            await db.commit()
    except Exception as e:
        logger.warning(f"Analysis run error: {e}", exc_info=True)
        run.status = "failed"
        run.errors = [str(e)]
        await db.commit()

    return {
        "id": str(run.id),
        "mandate_id": str(run.mandate_id),
        "status": run.status,
        "run_type": run.run_type,
        "agent1_status": run.agent1_status,
        "agent2_status": run.agent2_status,
        "agent3_status": run.agent3_status,
        "github_status": run.github_status,
        "agent4_status": run.agent4_status,
        "agent5_status": run.agent5_status,
        "errors": run.errors or [],
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


# ─── DELETE mandate ───────────────────────────────────────────────────────────

@router.delete("/{mandate_id}")
async def delete_mandate(
    mandate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Delete a mandate and all its associated evaluations and analysis runs."""
    try:
        mid = uuid_mod.UUID(mandate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mandate ID")

    result = await db.execute(
        select(Mandate).where(
            Mandate.id == mid,
            Mandate.recruiter_id == current_user["id"],
        )
    )
    mandate = result.scalar_one_or_none()
    if not mandate:
        raise HTTPException(status_code=404, detail="Mandate not found")

    # Cascade-delete evaluations
    evals = await db.execute(
        select(CandidateEvaluation).where(CandidateEvaluation.mandate_id == mid)
    )
    for ev in evals.scalars().all():
        await db.delete(ev)

    # Cascade-delete analysis runs
    runs = await db.execute(
        select(AnalysisRun).where(AnalysisRun.mandate_id == mid)
    )
    for run in runs.scalars().all():
        await db.delete(run)

    await db.delete(mandate)
    await db.commit()

    return {"deleted": mandate_id}
