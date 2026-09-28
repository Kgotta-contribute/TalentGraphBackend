import asyncio
import uuid
from app.db.session import AsyncSessionLocal
from app.models.mandate import Mandate
from sqlalchemy import select

async def update():
    async with AsyncSessionLocal() as s:
        res = await s.execute(select(Mandate).where(Mandate.id == uuid.UUID('b0000000-0000-0000-0000-000000000002')))
        m = res.scalar_one_or_none()
        if m:
            m.title = 'Full Stack Developer'
            m.company = 'Intuitive'
            if m.job_requirements:
                m.job_requirements['role'] = 'Full Stack Developer'
            await s.commit()
            print("Successfully updated b0000000 to Full Stack Developer (Intuitive)")

if __name__ == "__main__":
    asyncio.run(update())
