"""
Service layer for judgment list file creation and job management.
"""

import asyncio
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import aiofiles
from elasticsearch import Elasticsearch
from fastapi import BackgroundTasks, UploadFile
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config import settings
from src.training_data.models import (
    FeatureExtractionJob,
    FeatureExtractionJobCreateResponse,
    FeatureExtractionJobStatusResponse,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    FeatureSetRequest,
    FeatureSetResponse,
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    JudgmentListJob,
    ProductFeatures,
)
from src.training_data.repository import (
    FeatureExtractionJobRepository,
    JudgmentListJobRepository,
)


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

        # Create output filename with current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_judgment_list.csv"
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


class FeatureSetService:
    """Service for managing Elasticsearch featuresets and feature extraction."""

    def __init__(
        self, es_client: Elasticsearch, repository: FeatureExtractionJobRepository
    ):
        """
        Initialize the service with an Elasticsearch client and repository.

        Args:
            es_client: Elasticsearch client instance
            repository: Repository for feature extraction job persistence
        """
        self._es_client = es_client
        self._repository = repository

    async def create_featureset(
        self, featureset_name: str, request: FeatureSetRequest
    ) -> FeatureSetResponse:
        """
        Create a featureset in Elasticsearch Learning to Rank plugin.

        Args:
            featureset_name: Name of the featureset to create
            request: Request containing features configuration

        Returns:
            Response with featureset creation status

        Raises:
            Exception: If featureset creation fails
        """
        try:
            logger.info(
                f"Creating featureset '{featureset_name}' with {len(request.features)} features"
            )

            # Build featureset body for Elasticsearch LTR plugin
            featureset_body = {
                "featureset": {
                    "features": [
                        {
                            "name": feature.name,
                            "params": feature.params,
                            "template_language": feature.template_language,
                            "template": {
                                "lang": feature.template.lang,
                                "source": feature.template.source,
                            },
                        }
                        for feature in request.features
                    ],
                }
            }

            # Create featureset using Elasticsearch LTR API
            # The endpoint is: PUT _ltr/_featureset/{featureset_name}
            response = self._es_client.transport.perform_request(
                method="PUT",
                url=f"/_ltr/_featureset/{featureset_name}",
                body=featureset_body,
            )

            logger.info(
                f"Successfully created featureset '{featureset_name}'. Response: {response}"
            )

            return FeatureSetResponse(
                featureset_name=featureset_name,
                features_count=len(request.features),
                acknowledged=response.get("acknowledged", True),
            )

        except Exception as e:
            logger.error(f"Error creating featureset '{featureset_name}': {str(e)}")
            raise

    async def create_feature_extraction_job(
        self, request: FeatureExtractionRequest, background_tasks: BackgroundTasks
    ) -> FeatureExtractionJobCreateResponse:
        """
        Create a feature extraction job and schedule background processing.

        Args:
            request: Request containing judgment list filename and featureset name
            background_tasks: FastAPI background tasks for async processing

        Returns:
            Response with job ID and status

        Raises:
            FileNotFoundError: If judgment list file is not found
        """
        # Validate judgment list file exists
        judgment_list_path = Path(settings.OUTPUT_DIR) / request.judgment_list_filename

        if not judgment_list_path.exists():
            raise FileNotFoundError(
                f"Judgment list file not found: {request.judgment_list_filename}"
            )

        # Create job record
        job = FeatureExtractionJob(
            judgment_list_filename=request.judgment_list_filename,
            featureset_name=request.featureset_name,
            status=JobStatus.PENDING,
        )
        job = await self._repository.create(job)

        # Schedule background processing
        background_tasks.add_task(self._process_feature_extraction_job, job.id)
        logger.info(f"Scheduled feature extraction job {job.id}")

        return FeatureExtractionJobCreateResponse(
            job_id=job.id,
            judgment_list_filename=job.judgment_list_filename,
            featureset_name=job.featureset_name,
            status=job.status,
        )

    async def get_feature_extraction_job_status(
        self, job_id: UUID
    ) -> Optional[FeatureExtractionJobStatusResponse]:
        """
        Get the status of a feature extraction job.

        Args:
            job_id: The UUID of the job

        Returns:
            Job status response if found, None otherwise
        """
        job = await self._repository.get_by_id(job_id)
        if not job:
            return None

        return FeatureExtractionJobStatusResponse(
            job_id=job.id,
            judgment_list_filename=job.judgment_list_filename,
            featureset_name=job.featureset_name,
            status=job.status,
            output_file_path=job.output_file_path,
            error_message=job.error_message,
            total_products=job.total_products,
            products_with_features=job.products_with_features,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )

    async def _process_feature_extraction_job(self, job_id: UUID) -> None:
        """
        Process a feature extraction job in the background.

        Args:
            job_id: The UUID of the job to process
        """
        try:
            logger.info(f"Starting feature extraction for job {job_id}")

            # Update status to processing
            await self._repository.update_status(job_id, JobStatus.PROCESSING)

            # Get job details
            job = await self._repository.get_by_id(job_id)
            if not job:
                logger.error(f"Feature extraction job {job_id} not found")
                return

            # Extract features
            result = await self._extract_features_from_judgment_list(
                judgment_list_filename=job.judgment_list_filename,
                featureset_name=job.featureset_name,
            )

            # Save results to CSV
            output_path = await self._save_features_to_csv(
                result, job.judgment_list_filename, job.featureset_name
            )

            # Update job as completed
            await self._repository.update_status(
                job_id,
                JobStatus.COMPLETED,
                output_file_path=str(output_path),
                total_products=result.total_products,
                products_with_features=result.products_with_features,
            )

            logger.info(
                f"Successfully completed feature extraction job {job_id}. "
                f"Extracted features for {result.products_with_features}/{result.total_products} products"
            )

        except Exception as e:
            logger.error(f"Error processing feature extraction job {job_id}: {str(e)}")
            await self._repository.update_status(
                job_id, JobStatus.FAILED, error_message=str(e)
            )

    async def _extract_features_from_judgment_list(
        self, judgment_list_filename: str, featureset_name: str
    ) -> FeatureExtractionResponse:
        """
        Extract features for unique product IDs from a judgment list file.

        Uses asyncio with semaphore to run 5 concurrent requests to Elasticsearch.

        Args:
            judgment_list_filename: Name of the judgment list CSV file
            featureset_name: Name of the featureset to use

        Returns:
            Response with extracted features for each unique product

        Raises:
            FileNotFoundError: If judgment list file is not found
            Exception: If feature extraction fails
        """
        try:
            logger.info(
                f"Extracting features using featureset '{featureset_name}' "
                f"from judgment list '{judgment_list_filename}'"
            )

            # Construct full path to judgment list file
            judgment_list_path = Path(settings.OUTPUT_DIR) / judgment_list_filename

            if not judgment_list_path.exists():
                raise FileNotFoundError(
                    f"Judgment list file not found: {judgment_list_filename}"
                )

            # Read judgment list and extract unique product IDs
            unique_product_ids = await self._extract_unique_product_ids(
                judgment_list_path
            )

            logger.info(
                f"Found {len(unique_product_ids)} unique product IDs in judgment list"
            )

            # Create batches ahead of time to avoid duplication due to concurrency
            batch_size = 1000
            batches = []
            for i in range(0, len(unique_product_ids), batch_size):
                batch_ids = unique_product_ids[i : i + batch_size]
                batches.append(batch_ids)

            logger.info(f"Created {len(batches)} batches for concurrent processing")

            # Create semaphore to limit concurrent requests to 5
            semaphore = asyncio.Semaphore(5)

            # Create tasks for concurrent processing
            async def process_batch_with_semaphore(
                batch_ids: list[str], batch_number: int
            ) -> list[ProductFeatures]:
                """Process a single batch with semaphore control."""
                async with semaphore:
                    logger.debug(
                        f"Processing batch {batch_number}/{len(batches)} "
                        f"with {len(batch_ids)} product IDs"
                    )
                    return await self._extract_features_for_products(
                        product_ids=batch_ids, featureset_name=featureset_name
                    )

            # Create all tasks
            tasks = [
                process_batch_with_semaphore(batch, idx + 1)
                for idx, batch in enumerate(batches)
            ]

            # Execute all tasks concurrently with semaphore controlling concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect all successful results and log errors
            all_product_features = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing batch {idx + 1}: {str(result)}")
                else:
                    all_product_features.extend(result)

            logger.info(
                f"Successfully extracted features for {len(all_product_features)} products"
            )

            return FeatureExtractionResponse(
                featureset_name=featureset_name,
                total_products=len(unique_product_ids),
                products_with_features=len(all_product_features),
                product_features=all_product_features,
            )

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    async def _extract_unique_product_ids(self, judgment_list_path: Path) -> list[str]:
        """
        Extract unique product IDs from a judgment list CSV file.

        Args:
            judgment_list_path: Path to the judgment list CSV file

        Returns:
            List of unique product IDs as strings

        Raises:
            Exception: If CSV reading fails
        """
        unique_ids = set()

        async with aiofiles.open(judgment_list_path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(content))

        for row in csv_reader:
            if "product_id" in row and row["product_id"]:
                unique_ids.add(str(row["product_id"]))

        return list(unique_ids)

    async def _extract_features_for_products(
        self, product_ids: list[str], featureset_name: str
    ) -> list[ProductFeatures]:
        """
        Extract features for a batch of product IDs using Elasticsearch LTR.

        Uses asyncio.to_thread to run the synchronous Elasticsearch client in a thread.

        Args:
            product_ids: List of product IDs to extract features for
            featureset_name: Name of the featureset to use

        Returns:
            List of ProductFeatures with extracted feature vectors

        Raises:
            Exception: If Elasticsearch query fails
        """
        # Build the Elasticsearch query with LTR logging
        query_body = {
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"id": product_ids}},
                        {
                            "sltr": {
                                "_name": "logged_featureset",
                                "featureset": featureset_name,
                                "params": {},
                            }
                        },
                    ]
                }
            },
            "ext": {
                "ltr_log": {
                    "log_specs": {
                        "name": "log_entry1",
                        "named_query": "logged_featureset",
                    }
                }
            },
            "size": len(product_ids),
        }

        # Execute the query in a thread pool to avoid blocking
        response = await asyncio.to_thread(
            self._es_client.search, index="products", body=query_body
        )

        # Parse the response and extract features
        product_features_list = []

        for hit in response.get("hits", {}).get("hits", []):
            product_id = hit.get("_source", {}).get("id")

            # Extract LTR features from the response
            ltr_log = hit.get("fields", {}).get("_ltrlog", [])

            if ltr_log and len(ltr_log) > 0:
                # Parse the LTR log entry
                log_entry = ltr_log[0]
                if "log_entry1" in log_entry:
                    features = log_entry["log_entry1"]

                    product_features_list.append(
                        ProductFeatures(product_id=str(product_id), features=features)
                    )

        return product_features_list

    async def _save_features_to_csv(
        self,
        result: FeatureExtractionResponse,
        judgment_list_filename: str,
        featureset_name: str,
    ) -> Path:
        """
        Save extracted features to a CSV file.

        Args:
            result: Feature extraction results
            judgment_list_filename: Original judgment list filename
            featureset_name: Name of the featureset used

        Returns:
            Path to the created CSV file
        """
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_features_{featureset_name}.csv"
        output_path = Path(settings.OUTPUT_DIR) / output_filename

        logger.info(f"Saving features to CSV file: {output_path}")

        # Prepare CSV data
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

            # Write header
            if result.product_features:
                # Extract feature names from first product
                first_product = result.product_features[0]
                feature_names = [
                    feature.get("name", f"feature_{i + 1}")
                    for i, feature in enumerate(first_product.features)
                ]
                header = ["product_id"] + feature_names
                writer.writerow(header)

                logger.debug(f"CSV header with feature names: {header}")

                # Write data rows
                for product_feature in result.product_features:
                    row = [product_feature.product_id] + [
                        feature.get("value", "") for feature in product_feature.features
                    ]
                    writer.writerow(row)

            # Get the CSV content and write to file
            csv_content = output.getvalue()
            await f.write(csv_content)
            output.close()

        logger.info(
            f"Successfully saved {len(result.product_features)} feature vectors to {output_path}"
        )
        return output_path
