"""
FastAPI dependencies for training data module.
"""

from typing import AsyncGenerator

from elasticsearch import Elasticsearch
from fastapi import Depends
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config import settings
from src.database import app_engine, searchworker_engine
from src.training_data.repository import JudgmentListJobRepository
from src.training_data.service import FeatureSetService, JudgmentListService

# Elasticsearch client singleton
_es_client: Elasticsearch | None = None


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


def get_elasticsearch_client() -> Elasticsearch:
    """
    Dependency to get Elasticsearch client.

    Returns:
        Elasticsearch client instance

    Note:
        This creates a singleton Elasticsearch client to avoid creating
        multiple connections for each request.
    """
    global _es_client

    if _es_client is None:
        # Build Elasticsearch URL
        es_url = f"{settings.ELASTIC_HOST}:{settings.ELASTIC_PORT}"

        # Create Elasticsearch client with authentication if credentials provided
        if settings.ELASTIC_USER and settings.ELASTIC_PASSWORD:
            _es_client = Elasticsearch(
                [es_url],
                http_auth=(settings.ELASTIC_USER, settings.ELASTIC_PASSWORD),
                basic_auth=(settings.ELASTIC_USER, settings.ELASTIC_PASSWORD),
                timeout=300,
            )
            logger.info(f"Created Elasticsearch client with authentication to {es_url}")
        else:
            _es_client = Elasticsearch([es_url])
            logger.info(
                f"Created Elasticsearch client without authentication to {es_url}"
            )

    return _es_client


def get_featureset_service(
    es_client: Elasticsearch = Depends(get_elasticsearch_client),
) -> FeatureSetService:
    """
    Dependency to get featureset service.

    Args:
        es_client: Elasticsearch client from get_elasticsearch_client dependency

    Returns:
        FeatureSetService instance
    """
    return FeatureSetService(es_client)
