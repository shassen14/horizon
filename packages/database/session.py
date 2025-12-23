# packages/database/session.py

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from sqlalchemy.orm import sessionmaker

from packages.quant_lib.config import settings

# Create an async engine
engine: AsyncEngine = create_async_engine(settings.db.URL, echo=False)

# Create a session maker
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db_session():
    """Provides a transactional scope around a series of operations."""
    session = AsyncSessionFactory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_autocommit_connection():
    """
    Yields a raw SQLAlchemy AsyncConnection in AUTOCOMMIT mode.
    Crucial for administrative tasks like VACUUM and CREATE DATABASE.
    """
    # Create a fresh engine specifically for this task
    # We use isolation_level="AUTOCOMMIT" in the engine arguments
    maintenance_engine = create_async_engine(
        settings.db.URL, isolation_level="AUTOCOMMIT", echo=False
    )

    try:
        async with maintenance_engine.connect() as conn:
            yield conn
    finally:
        await maintenance_engine.dispose()
