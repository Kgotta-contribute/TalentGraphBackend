import uuid as uuid_mod
import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user
from app.models.candidate import Candidate
from app.models.evaluation import CandidateEvaluation
from app.agents.resume_parser import parse_resume
from app.agents.requirement_verifier import verify_requirements
from app.agents.report_generator import generate_report
import logging
from app.mcp.github_client import GitHubMCPClient

logger = logging.getLogger("talent_agent.candidates")

router = APIRouter()


class ImportFromResumeBody(BaseModel):
    resume_id: str
    resume_text: str
    puter_file_path: str | None = None
    resumeiq_audit_json: dict | None = None
    company_name: str | None = None
    job_title: str | None = None
    mandate_id: str | None = None


class GenerateReportBody(BaseModel):
    mandate_id: str


def candidate_to_dict(c: Candidate) -> dict:
    return {
        "id": str(c.id),
        "full_name": c.full_name,
        "email": c.email,
        "github_url": c.github_url,
        "linkedin_url": c.linkedin_url,
        "current_title": c.current_title,
        "years_experience": c.years_experience,
        "profile": c.profile,
        "resumeiq_resume_id": c.resumeiq_resume_id,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


@router.get("")
async def list_all_candidates(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all candidates in the system."""
    result = await db.execute(select(Candidate).order_by(Candidate.created_at.desc()))
    candidates = result.scalars().all()
    return [candidate_to_dict(c) for c in candidates]



@router.post("/import-from-resume")
async def import_candidate(
    body: ImportFromResumeBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Import a candidate from an existing ResumeIQ report.
    Reads already-extracted text + metadata. Runs Agent 2 (resume parser).
    Creates a Candidate record. Optionally links to a mandate as an evaluation.
    """
    cleaned_text = (body.resume_text or "").replace("\x00", " ")
    if not cleaned_text.strip() or len(cleaned_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Cannot import candidate with empty or unreadable resume text. Only valid PDF, DOC, DOCX, and TXT files with readable text content are supported.",
        )

    # Check for duplicate import of the exact same ResumeIQ record
    if body.resume_id:
        existing_res = await db.execute(
            select(Candidate).where(Candidate.resumeiq_resume_id == body.resume_id)
        )
        existing_cand = existing_res.scalar_one_or_none()
        if existing_cand:
            if body.mandate_id:
                eval_check = await db.execute(
                    select(CandidateEvaluation).where(
                        CandidateEvaluation.mandate_id == uuid_mod.UUID(body.mandate_id),
                        CandidateEvaluation.candidate_id == existing_cand.id,
                    )
                )
                if not eval_check.scalar_one_or_none():
                    db.add(CandidateEvaluation(
                        mandate_id=uuid_mod.UUID(body.mandate_id),
                        candidate_id=existing_cand.id,
                        status="parsed",
                    ))
                    await db.commit()
            return {"candidate_id": str(existing_cand.id), "status": "existing"}

    # Enforce MAX 500 candidates retention limit (FIFO auto-prune oldest)
    from sqlalchemy import func
    total_count = await db.scalar(select(func.count(Candidate.id)))
    if total_count and total_count >= 500:
        excess = total_count - 500 + 1
        oldest_res = await db.execute(
            select(Candidate.id).order_by(Candidate.created_at.asc()).limit(excess)
        )
        oldest_ids = oldest_res.scalars().all()
        if oldest_ids:
            await db.execute(
                CandidateEvaluation.__table__.delete().where(CandidateEvaluation.candidate_id.in_(oldest_ids))
            )
            try:
                from app.models.document import DocumentChunk
                await db.execute(
                    DocumentChunk.__table__.delete().where(DocumentChunk.candidate_id.in_(oldest_ids))
                )
            except Exception:
                pass
            await db.execute(
                Candidate.__table__.delete().where(Candidate.id.in_(oldest_ids))
            )
            await db.commit()
            logger.info(f"Auto-pruned {len(oldest_ids)} oldest candidate records to respect 500-resume quota")

    # Run Agent 2 to extract structured profile from resume text
    try:
        profile = await parse_resume(cleaned_text)
    except Exception as e:
        print(f"Agent 2 parsing fallback: {e}")
        profile = None

    candidate = Candidate(
        resumeiq_resume_id=body.resume_id,
        puter_file_path=body.puter_file_path,
        raw_text=cleaned_text[:50000],  # safeguard length
        resumeiq_audit_json=body.resumeiq_audit_json,
        profile=profile.model_dump() if profile else None,
        full_name=profile.full_name if profile and profile.full_name else body.company_name or "Parsed Candidate",
        email=profile.email if profile else None,
        github_url=profile.github_url if profile else None,
        linkedin_url=profile.linkedin_url if profile else None,
        current_title=profile.current_title if profile and profile.current_title else body.job_title or "Software Engineer",
        years_experience=str(profile.years_experience) if profile else "3.0",
    )
    db.add(candidate)
    await db.commit()
    await db.refresh(candidate)

    # Optionally link to a mandate
    if body.mandate_id:
        try:
            evaluation = CandidateEvaluation(
                mandate_id=uuid_mod.UUID(body.mandate_id),
                candidate_id=candidate.id,
                status="parsed",
            )
            db.add(evaluation)
            await db.commit()
        except Exception as eval_err:
            print(f"Evaluation link error: {eval_err}")

    return {"candidate_id": str(candidate.id), "status": "created"}


@router.get("/{candidate_id}")
async def get_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(select(Candidate).where(Candidate.id == uuid_mod.UUID(candidate_id)))
    cand = result.scalar_one_or_none()
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate_to_dict(cand)


@router.delete("/{candidate_id}")
async def delete_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    cid = uuid_mod.UUID(candidate_id)
    # delete evaluations first
    await db.execute(
        CandidateEvaluation.__table__.delete().where(CandidateEvaluation.candidate_id == cid)
    )
    # delete document chunks if any
    try:
        from app.models.document import DocumentChunk
        await db.execute(
            DocumentChunk.__table__.delete().where(DocumentChunk.candidate_id == cid)
        )
    except Exception as e:
        logger.warning(f"Could not delete document chunks for candidate {cid}: {e}", exc_info=True)
    # delete candidate
    await db.execute(
        Candidate.__table__.delete().where(Candidate.id == cid)
    )
    await db.commit()
    return {"message": f"Candidate {candidate_id} deleted"}



@router.post("/{candidate_id}/github-analysis")
async def trigger_github(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Trigger GitHub MCP analysis for this candidate (only if github_url is present)."""
    result = await db.execute(select(Candidate).where(Candidate.id == uuid_mod.UUID(candidate_id)))
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if not candidate.github_url:
        raise HTTPException(status_code=400, detail="Candidate has no GitHub URL")

    # Extract GitHub username from URL
    match = re.search(r'github\.com/([^/\s]+)', candidate.github_url)
    if not match:
        raise HTTPException(status_code=400, detail="Could not parse GitHub username from URL")
    username = match.group(1)

    try:
        client = GitHubMCPClient()
        repos = await client.get_user_repos(username)

        # Analyze top 5 repos
        analyzed_repos = []
        verified_skills = []
        all_languages = set()

        for repo in repos[:5]:
            repo_name = repo.get("name", "")
            is_fork = repo.get("fork", False)
            analyzed_repos.append(repo_name)

            try:
                langs = await client.get_repo_languages(username, repo_name)
                all_languages.update(langs.keys())
            except Exception as e:
                logger.warning(f"Failed to fetch repo languages for {username}/{repo_name}: {e}", exc_info=True)

        # Skills verified from language data
        lang_to_skill = {
            "Python": "Python", "TypeScript": "TypeScript", "JavaScript": "JavaScript",
            "Java": "Java", "Go": "Go", "Rust": "Rust", "C++": "C++", "C#": "C#",
            "Ruby": "Ruby", "PHP": "PHP", "Swift": "Swift", "Kotlin": "Kotlin"
        }
        verified_skills = [lang_to_skill[l] for l in all_languages if l in lang_to_skill]

        # Activity signal
        public_repos = repos
        forked = [r for r in public_repos if r.get("fork")]
        if len(forked) == len(public_repos) and len(public_repos) > 0:
            activity_signal = "forks_only"
        elif len(public_repos) == 0:
            activity_signal = "unavailable"
        else:
            activity_signal = "active"

        analysis = {
            "candidate_id": candidate_id,
            "github_url": candidate.github_url,
            "repos_analyzed": analyzed_repos,
            "verified_skills": verified_skills,
            "unverified_skills": [],
            "activity_signal": activity_signal,
            "evidence": [
                {
                    "skill": skill,
                    "verified": True,
                    "repo": analyzed_repos[0] if analyzed_repos else "",
                    "evidence": f"Found in repository languages",
                    "confidence": 0.75,
                }
                for skill in verified_skills[:5]
            ],
            "summary": f"Analyzed {len(analyzed_repos)} repositories. Detected languages: {', '.join(list(all_languages)[:8]) if all_languages else 'none'}. Activity signal: {activity_signal}.",
        }

        # Store in candidate evaluations (update all evals for this candidate)
        eval_result = await db.execute(
            select(CandidateEvaluation).where(CandidateEvaluation.candidate_id == candidate.id)
        )
        evals = eval_result.scalars().all()
        for ev in evals:
            ev.github_analysis = analysis
            ev.github_verified = len(verified_skills) > 0
        if evals:
            await db.commit()

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub analysis failed: {str(e)}")


@router.get("/{candidate_id}/github-analysis")
async def get_github(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = await db.execute(
        select(CandidateEvaluation).where(
            CandidateEvaluation.candidate_id == uuid_mod.UUID(candidate_id)
        )
    )
    eval_ = result.scalars().first()
    if not eval_ or not eval_.github_analysis:
        raise HTTPException(status_code=404, detail="No GitHub analysis found")
    return eval_.github_analysis


@router.post("/{candidate_id}/report")
async def generate_report_route(
    candidate_id: str,
    body: GenerateReportBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Agent 5: Generate recruitment dossier for a candidate+mandate pair."""
    cand_result = await db.execute(
        select(Candidate).where(Candidate.id == uuid_mod.UUID(candidate_id))
    )
    candidate = cand_result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    eval_result = await db.execute(
        select(CandidateEvaluation).where(
            CandidateEvaluation.candidate_id == uuid_mod.UUID(candidate_id),
            CandidateEvaluation.mandate_id == uuid_mod.UUID(body.mandate_id),
        )
    )
    evaluation = eval_result.scalar_one_or_none()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation for this mandate not found")

    try:
        dossier = await generate_report(
            candidate_id=candidate_id,
            candidate_profile=candidate.profile or {},
            verification_result=evaluation.verification_result or {},
            scoring_result={
                "final_score": evaluation.final_score,
                "tier": evaluation.tier,
                "score_technical": evaluation.score_technical,
                "score_experience": evaluation.score_experience,
            },
            github_analysis=evaluation.github_analysis,
        )
        evaluation.report = dossier.model_dump()
        await db.commit()
        return evaluation.report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/{candidate_id}/report")
async def get_report_route(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    eval_result = await db.execute(
        select(CandidateEvaluation).where(
            CandidateEvaluation.candidate_id == uuid_mod.UUID(candidate_id)
        )
    )
    eval_ = eval_result.scalars().first()
    if not eval_ or not eval_.report:
        raise HTTPException(status_code=404, detail="No report generated yet")
    return eval_.report
