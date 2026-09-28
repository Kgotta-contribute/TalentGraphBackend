from fastapi import Header, HTTPException
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

async def get_current_user(x_puter_token: str | None = Header(default=None)):
    if settings.auth_mode == "dev":
        return {"id": "dev-recruiter", "username": "Developer"}
    # Production: verify Puter token here
    if not x_puter_token:
        raise HTTPException(status_code=401, detail="Missing Puter auth token")
    # TODO: verify token with Puter API
    return {"id": x_puter_token, "username": "Recruiter"}
