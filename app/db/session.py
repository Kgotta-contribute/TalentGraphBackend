from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

db_url = (settings.database_url or "").strip()
if not db_url:
    db_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
