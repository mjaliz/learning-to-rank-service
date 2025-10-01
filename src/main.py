"""Main entry point for the learning-to-rank service."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from sqlmodel import SQLModel

from src.config import settings
from src.database import app_engine, searchworker_engine
from src.training_data.router import router as training_data_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting learning-to-rank-service")
    logger.info(
        f"App PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    logger.info(
        f"Searchworker PostgreSQL: {settings.POSTGRES_SEARCHWORKER_HOST}:{settings.POSTGRES_SEARCHWORKER_PORT}/{settings.POSTGRES_SEARCHWORKER_DB}"
    )
    logger.info(
        f"Elasticsearch: {settings.ELASTIC_HOST}:{settings.ELASTIC_PORT}/{settings.ELASTIC_INDEX}"
    )

    # Create database tables in app database
    async with app_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables created successfully in app database")

    yield

    # Shutdown
    logger.info("Shutting down learning-to-rank-service")
    await app_engine.dispose()
    await searchworker_engine.dispose()


# Create FastAPI application
app = FastAPI(
    title="Learning to Rank Service",
    description="Service for creating and managing learning-to-rank training data",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(training_data_router)


@app.get("/", tags=["health"])
async def root():
    """Root endpoint for health check."""
    return {
        "service": "learning-to-rank-service",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def main() -> None:
    """Main application entry point."""
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
