from typing import TypedDict

class TalentAgentState(TypedDict):
    run_id: str
    mandate_id: str
    candidate_ids: list[str]
    current_candidate_id: str | None
    job_requirements_id: str | None
    verification_result_ids: dict[str, str]
    github_analysis_ids: dict[str, str]
    ranking_result_ids: dict[str, str]
    report_id: str | None
    run_status: str
    errors: list[str]
    current_candidate_github_url: str | None
