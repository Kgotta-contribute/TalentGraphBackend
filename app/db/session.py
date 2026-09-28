import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

logger = logging.getLogger("talent_agent.db")

db_url = (settings.database_url or "").strip()
while (db_url.startswith('"') and db_url.endswith('"')) or (db_url.startswith("'") and db_url.endswith("'")):
    db_url = db_url[1:-1].strip()

fallback_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

if not db_url:
    db_url = fallback_url

try:
    engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
except Exception as e:
    logger.error(f"Failed to initialize database engine with URL '{db_url}': {e}. Using fallback engine.")
    engine = create_async_engine(fallback_url, echo=False)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
