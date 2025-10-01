"""
Database engine configuration and management.

This module provides two separate async database engines:
1. app_engine: For the application's own database (creates tables, manages data)
2. searchworker_engine: For read-only queries against the searchworker database
"""

from urllib.parse import quote_plus

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config import settings


def create_app_engine() -> AsyncEngine:
    """
    Create async database engine for the application's database.

    This engine is used for creating tables and managing application data.

    Returns:
        AsyncEngine instance for the application database
    """
    database_url = (
        f"postgresql+asyncpg://{settings.POSTGRES_USER}:{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    engine = create_async_engine(
        database_url,
        echo=True,
        future=True,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

    logger.info(
        f"Created app database engine: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    return engine


def create_searchworker_engine() -> AsyncEngine:
    """
    Create async database engine for the searchworker database.

    This engine is used for read-only queries against the searchworker database.

    Returns:
        AsyncEngine instance for the searchworker database
    """
    database_url = (
        f"postgresql+asyncpg://{settings.POSTGRES_SEARCHWORKER_USER}:{quote_plus(settings.POSTGRES_SEARCHWORKER_PASSWORD)}"
        f"@{settings.POSTGRES_SEARCHWORKER_HOST}:{settings.POSTGRES_SEARCHWORKER_PORT}/{settings.POSTGRES_SEARCHWORKER_DB}"
    )

    engine = create_async_engine(
        database_url,
        echo=True,
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    logger.info(
        f"Created searchworker database engine: {settings.POSTGRES_SEARCHWORKER_HOST}:{settings.POSTGRES_SEARCHWORKER_PORT}/{settings.POSTGRES_SEARCHWORKER_DB}"
    )

    return engine


# Create singleton engine instances
app_engine = create_app_engine()
searchworker_engine = create_searchworker_engine()
