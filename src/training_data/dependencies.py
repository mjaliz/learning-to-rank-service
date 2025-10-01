"""
FastAPI dependencies for training data module.
"""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from src.database import app_engine, searchworker_engine
from src.training_data.repository import JudgmentListJobRepository
from src.training_data.service import JudgmentListService


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session for the application database.

    Yields:
        AsyncSession for database operations
    """
    async with AsyncSession(app_engine) as session:
        yield session


async def get_searchworker_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session for the searchworker database.

    This should be used for read-only queries against the searchworker database.

    Yields:
        AsyncSession for searchworker database operations
    """
    async with AsyncSession(searchworker_engine) as session:
        yield session


def get_searchworker_engine() -> AsyncEngine:
    """
    Dependency to get the searchworker database engine.

    Returns:
        AsyncEngine for searchworker database
    """
    return searchworker_engine


async def get_repository(
    session: AsyncSession = Depends(get_session),
) -> JudgmentListJobRepository:
    """
    Dependency to get job repository.

    Args:
        session: Database session from get_session dependency

    Returns:
        Repository instance
    """
    return JudgmentListJobRepository(session)


async def get_service(
    repository: JudgmentListJobRepository = Depends(get_repository),
    searchworker_engine: AsyncEngine = Depends(get_searchworker_engine),
) -> JudgmentListService:
    """
    Dependency to get judgment list service.

    Args:
        repository: Repository instance from get_repository dependency
        searchworker_engine: Searchworker database engine from get_searchworker_engine dependency

    Returns:
        Service instance
    """
    return JudgmentListService(repository, searchworker_engine)
