"""
Repository layer for judgment list job persistence.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.training_data.models import (
    FeatureExtractionJob,
    JobStatus,
    JudgmentListJob,
)


class JudgmentListJobRepository:
    """Repository for managing judgment list job persistence."""

    def __init__(self, session: AsyncSession):
        """
        Initialize the repository with a database session.

        Args:
            session: AsyncSession for database operations
        """
        self._session = session

    async def create(self, job: JudgmentListJob) -> JudgmentListJob:
        """
        Create a new job in the database.

        Args:
            job: The job to create

        Returns:
            The created job with ID populated
        """
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        logger.info(f"Created new job with ID: {job.id}")
        return job

    async def get_by_id(self, job_id: UUID) -> Optional[JudgmentListJob]:
        """
        Retrieve a job by its ID.

        Args:
            job_id: The UUID of the job to retrieve

        Returns:
            The job if found, None otherwise
        """
        statement = select(JudgmentListJob).where(JudgmentListJob.id == job_id)
        result = await self._session.execute(statement)
        job = result.scalar_one_or_none()

        if job:
            logger.debug(f"Retrieved job {job_id} with status {job.status}")
        else:
            logger.warning(f"Job {job_id} not found")

        return job

    async def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        error_message: Optional[str] = None,
        output_file_path: Optional[str] = None,
    ) -> Optional[JudgmentListJob]:
        """
        Update the status of a job.

        Args:
            job_id: The UUID of the job to update
            status: The new status
            error_message: Optional error message if the job failed
            output_file_path: Optional path to the output file if completed

        Returns:
            The updated job if found, None otherwise
        """
        job = await self.get_by_id(job_id)
        if not job:
            return None

        job.status = status

        if status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = datetime.utcnow()

        if error_message:
            job.error_message = error_message

        if output_file_path:
            job.output_file_path = output_file_path

        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)

        logger.info(f"Updated job {job_id} status to {status}")
        return job

    async def list_jobs(
        self, status: Optional[JobStatus] = None, limit: int = 100
    ) -> list[JudgmentListJob]:
        """
        List jobs, optionally filtered by status.

        Args:
            status: Optional status to filter by
            limit: Maximum number of jobs to return

        Returns:
            List of jobs matching the criteria
        """
        statement = select(JudgmentListJob)

        if status:
            statement = statement.where(JudgmentListJob.status == status)

        statement = statement.order_by(JudgmentListJob.created_at.desc()).limit(limit)

        result = await self._session.execute(statement)
        jobs = result.scalars().all()

        logger.debug(f"Retrieved {len(jobs)} jobs")
        return list(jobs)


class FeatureExtractionJobRepository:
    """Repository for managing feature extraction job persistence."""

    def __init__(self, session: AsyncSession):
        """
        Initialize the repository with a database session.

        Args:
            session: AsyncSession for database operations
        """
        self._session = session

    async def create(self, job: FeatureExtractionJob) -> FeatureExtractionJob:
        """
        Create a new feature extraction job in the database.

        Args:
            job: The job to create

        Returns:
            The created job with ID populated
        """
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        logger.info(f"Created new feature extraction job with ID: {job.id}")
        return job

    async def get_by_id(self, job_id: UUID) -> Optional[FeatureExtractionJob]:
        """
        Retrieve a feature extraction job by its ID.

        Args:
            job_id: The UUID of the job to retrieve

        Returns:
            The job if found, None otherwise
        """
        statement = select(FeatureExtractionJob).where(
            FeatureExtractionJob.id == job_id
        )
        result = await self._session.execute(statement)
        job = result.scalar_one_or_none()

        if job:
            logger.debug(
                f"Retrieved feature extraction job {job_id} with status {job.status}"
            )
        else:
            logger.warning(f"Feature extraction job {job_id} not found")

        return job

    async def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        error_message: Optional[str] = None,
        output_file_path: Optional[str] = None,
        total_products: Optional[int] = None,
        products_with_features: Optional[int] = None,
    ) -> Optional[FeatureExtractionJob]:
        """
        Update the status of a feature extraction job.

        Args:
            job_id: The UUID of the job to update
            status: The new status
            error_message: Optional error message if the job failed
            output_file_path: Optional path to the output file if completed
            total_products: Total number of products processed
            products_with_features: Number of products with extracted features

        Returns:
            The updated job if found, None otherwise
        """
        job = await self.get_by_id(job_id)
        if not job:
            return None

        job.status = status

        if status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = datetime.utcnow()

        if error_message:
            job.error_message = error_message

        if output_file_path:
            job.output_file_path = output_file_path

        if total_products is not None:
            job.total_products = total_products

        if products_with_features is not None:
            job.products_with_features = products_with_features

        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)

        logger.info(f"Updated feature extraction job {job_id} status to {status}")
        return job
