import uuid as uuid_mod
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.deps import get_db
from app.models.analysis import AnalysisRun

router = APIRouter()


@router.get("/{run_id}/events")
async def stream_run_events(run_id: str, db: AsyncSession = Depends(get_db)):
    """SSE streaming endpoint for real-time analysis run updates."""
    async def event_generator():
        for _ in range(30):  # Poll up to 30 times (30 seconds)
            try:
                result = await db.execute(
                    select(AnalysisRun).where(AnalysisRun.id == uuid_mod.UUID(run_id))
                )
                run = result.scalar_one_or_none()
                if run:
                    payload = {
                        "id": str(run.id),
                        "mandate_id": str(run.mandate_id) if run.mandate_id else None,
                        "status": run.status,
                        "agent1_status": run.agent1_status,
                        "agent2_status": run.agent2_status,
                        "agent3_status": run.agent3_status,
                        "github_status": run.github_status,
                        "agent4_status": run.agent4_status,
                        "agent5_status": run.agent5_status,
                        "errors": run.errors or [],
                    }
                    yield {"event": "status", "data": json.dumps(payload)}

                    if run.status in ["completed", "failed"]:
                        break
                else:
                    yield {"event": "status", "data": json.dumps({"status": "running"})}
            except Exception as ex:
                yield {"event": "error", "data": json.dumps({"error": str(ex)})}
                break

            await asyncio.sleep(1)

        yield {"event": "done", "data": json.dumps({"completed": True})}

    return EventSourceResponse(event_generator())
