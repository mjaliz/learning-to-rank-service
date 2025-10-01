"""
Service layer for judgment list file creation and job management.
"""

import csv
import io
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import aiofiles
from fastapi import BackgroundTasks, UploadFile
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config import settings
from src.training_data.models import (
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    JudgmentListJob,
)
from src.training_data.repository import JudgmentListJobRepository


class JudgmentListService:
    """Service for managing judgment list file creation jobs."""

    def __init__(
        self, repository: JudgmentListJobRepository, searchworker_engine: AsyncEngine
    ):
        """
        Initialize the service with a repository and searchworker database engine.

        Args:
            repository: Repository for job persistence
            searchworker_engine: Async engine for searchworker database queries
        """
        self._repository = repository
        self._searchworker_engine = searchworker_engine
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure upload and output directories exist."""
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured upload and output directories exist")

    async def create_job_from_upload(
        self,
        file: UploadFile,
        background_tasks: BackgroundTasks,
    ) -> JobCreateResponse:
        """
        Create a new job from an uploaded SQL file.

        Args:
            file: The uploaded SQL file
            background_tasks: FastAPI background tasks for async processing

        Returns:
            Response containing job information

        Raises:
            ValueError: If the file is invalid or too large
        """
        # Validate file
        if not file.filename:
            raise ValueError("Filename is required")

        if not file.filename.endswith(".sql"):
            raise ValueError("Only SQL files are allowed")

        # Read and validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise ValueError(
                f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )

        # Save uploaded file
        upload_path = Path(settings.UPLOAD_DIR) / f"{uuid4().hex}_{file.filename}"
        async with aiofiles.open(upload_path, "wb") as f:
            await f.write(content)

        logger.info(f"Saved uploaded file to {upload_path}")

        # Create job record
        job = JudgmentListJob(
            filename=file.filename,
            sql_file_path=str(upload_path),
            status=JobStatus.PENDING,
        )
        job = await self._repository.create(job)

        # Schedule background processing
        background_tasks.add_task(self._process_job, job.id)
        logger.info(f"Scheduled background processing for job {job.id}")

        return JobCreateResponse(
            job_id=job.id,
            filename=job.filename,
            status=job.status,
        )

    async def get_job_status(self, job_id: UUID) -> Optional[JobStatusResponse]:
        """
        Get the status of a job.

        Args:
            job_id: The UUID of the job

        Returns:
            Job status response if found, None otherwise
        """
        job = await self._repository.get_by_id(job_id)
        if not job:
            return None

        return JobStatusResponse(
            job_id=job.id,
            filename=job.filename,
            status=job.status,
            output_file_path=job.output_file_path,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )

    async def _process_job(self, job_id: UUID) -> None:
        """
        Process a job to create a judgment list file.

        This is the background task that processes the SQL file.

        Args:
            job_id: The UUID of the job to process
        """
        try:
            logger.info(f"Starting processing for job {job_id}")

            # Update status to processing
            await self._repository.update_status(job_id, JobStatus.PROCESSING)

            # Get job details
            job = await self._repository.get_by_id(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return

            # Process the SQL file
            output_path = await self._create_judgment_list(
                job.sql_file_path, job.filename
            )

            # Update job as completed
            await self._repository.update_status(
                job_id,
                JobStatus.COMPLETED,
                output_file_path=str(output_path),
            )

            logger.info(f"Successfully completed job {job_id}")

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            await self._repository.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(e),
            )

    async def _create_judgment_list(
        self, sql_file_path: str, original_filename: str
    ) -> Path:
        """
        Create a judgment list file from a SQL file.

        Executes the SQL query against the searchworker database and saves
        the results as a CSV file.

        Args:
            sql_file_path: Path to the SQL file
            original_filename: Original filename of the uploaded file

        Returns:
            Path to the created judgment list CSV file
        """
        # Read SQL file
        async with aiofiles.open(sql_file_path, "r") as f:
            sql_content = await f.read()

        logger.debug(f"Read SQL file with {len(sql_content)} characters")

        # Create output filename
        output_filename = original_filename.replace(".sql", "_judgment_list.csv")
        output_path = Path(settings.OUTPUT_DIR) / output_filename

        # Execute SQL and save results as CSV
        await self._process_sql_to_judgment_list(sql_content, output_path)

        logger.info(f"Created judgment list CSV file at {output_path}")
        return output_path

    async def _process_sql_to_judgment_list(
        self, sql_content: str, output_path: Path
    ) -> None:
        """
        Execute SQL query against searchworker database and save results as CSV.

        Args:
            sql_content: SQL query content from the uploaded file
            output_path: Path where the CSV file should be saved

        Raises:
            Exception: If SQL execution or CSV writing fails
        """
        logger.info("Executing SQL query against searchworker database")

        # Create a session for the searchworker database
        async with AsyncSession(self._searchworker_engine) as session:
            try:
                # Execute the SQL query (use execute for raw SQL, not exec)
                result = await session.execute(text(sql_content))

                # Fetch all rows
                rows = result.all()
                logger.info(f"Query returned {len(rows)} rows")

                # Get column names from the result (use keys() for CursorResult)
                column_names = list(result.keys())
                logger.debug(f"Column names: {column_names}")

                # Write results to CSV file
                async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                    # Use csv module through StringIO for proper CSV formatting
                    # Create in-memory string buffer
                    output = io.StringIO()
                    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

                    # Write header
                    writer.writerow(column_names)

                    # Write data rows
                    for row in rows:
                        # Convert row to list, handling None values
                        row_values = [
                            value if value is not None else "" for value in row
                        ]
                        writer.writerow(row_values)

                    # Get the CSV content and write to file
                    csv_content = output.getvalue()
                    await f.write(csv_content)
                    output.close()

                logger.info(f"Successfully wrote {len(rows)} rows to CSV file")

            except Exception as e:
                logger.error(f"Error executing SQL query or writing CSV: {str(e)}")
                raise
