"""
API router for judgment list file creation endpoints.
"""

from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    HTTPException,
    Path,
    UploadFile,
    status,
)
from loguru import logger

from src.training_data.dependencies import get_featureset_service, get_service
from src.training_data.models import (
    FeatureExtractionJobCreateResponse,
    FeatureExtractionJobStatusResponse,
    FeatureExtractionRequest,
    FeatureSetRequest,
    FeatureSetResponse,
    JobCreateResponse,
    JobStatusResponse,
    TrainingDataMergeRequest,
    TrainingDataMergeResponse,
)
from src.training_data.service import FeatureSetService, JudgmentListService

router = APIRouter(
    prefix="/api/v1/training-data",
    tags=["training-data"],
)


@router.post(
    "/judgment-list/upload",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload SQL file for judgment list creation",
    description="Upload a SQL file to create a judgment list. The file will be processed asynchronously.",
)
async def upload_sql_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="SQL file to process"),
    service: JudgmentListService = Depends(get_service),
) -> JobCreateResponse:
    """
    Upload a SQL file to create a judgment list.

    Args:
        background_tasks: FastAPI background tasks
        file: The uploaded SQL file
        service: Judgment list service dependency

    Returns:
        Job creation response with job ID and status

    Raises:
        HTTPException: If file validation fails
    """
    try:
        logger.info(f"Received file upload request: {file.filename}")

        response = await service.create_job_from_upload(file, background_tasks)

        logger.info(f"Created job {response.job_id} for file {file.filename}")
        return response

    except ValueError as e:
        logger.warning(f"File validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job. Please try again later.",
        )


@router.get(
    "/judgment-list/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check judgment list creation status",
    description="Check the status of a judgment list creation job by its ID.",
)
async def get_job_status(
    job_id: UUID,
    service: JudgmentListService = Depends(get_service),
) -> JobStatusResponse:
    """
    Get the status of a judgment list creation job.

    Args:
        job_id: The UUID of the job
        service: Judgment list service dependency

    Returns:
        Job status response with current status and details

    Raises:
        HTTPException: If job is not found
    """
    try:
        logger.debug(f"Checking status for job {job_id}")

        job_status = await service.get_job_status(job_id)

        if not job_status:
            logger.warning(f"Job {job_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job with ID {job_id} not found",
            )

        logger.debug(f"Job {job_id} status: {job_status.status}")
        return job_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status. Please try again later.",
        )


@router.put(
    "/featureset/{featureset_name}",
    response_model=FeatureSetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Elasticsearch featureset",
    description="Create a featureset in Elasticsearch with the specified features for Learning to Rank.",
)
async def create_featureset(
    featureset_name: str = Path(
        ..., description="Name of the featureset to create", min_length=1
    ),
    request: FeatureSetRequest = Body(
        ..., description="Featureset configuration with features"
    ),
    service: FeatureSetService = Depends(get_featureset_service),
) -> FeatureSetResponse:
    """
    Create a featureset in Elasticsearch Learning to Rank plugin.

    Args:
        featureset_name: Name of the featureset to create
        request: Request body containing features configuration
        service: Featureset service dependency

    Returns:
        Featureset creation response with status

    Raises:
        HTTPException: If featureset creation fails
    """
    try:
        logger.info(
            f"Received request to create featureset '{featureset_name}' with {len(request.features)} features"
        )

        response = await service.create_featureset(featureset_name, request)

        logger.info(f"Successfully created featureset '{featureset_name}'")
        return response

    except Exception as e:
        logger.error(f"Error creating featureset '{featureset_name}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create featureset: {str(e)}",
        )


@router.post(
    "/featureset/extract-features",
    response_model=FeatureExtractionJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Extract features from judgment list",
    description="Create a job to extract features for unique product IDs from a judgment list file using a specified featureset.",
)
async def extract_features(
    background_tasks: BackgroundTasks,
    request: FeatureExtractionRequest = Body(
        ..., description="Feature extraction configuration"
    ),
    service: FeatureSetService = Depends(get_featureset_service),
) -> FeatureExtractionJobCreateResponse:
    """
    Create a feature extraction job to extract features for unique product IDs.

    Args:
        background_tasks: FastAPI background tasks
        request: Request body containing judgment list filename and featureset name
        service: Featureset service dependency

    Returns:
        Job creation response with job ID and status

    Raises:
        HTTPException: If file not found or job creation fails
    """
    try:
        logger.info(
            f"Received request to extract features using featureset '{request.featureset_name}' "
            f"from judgment list '{request.judgment_list_filename}'"
        )

        response = await service.create_feature_extraction_job(
            request, background_tasks
        )

        logger.info(f"Created feature extraction job {response.job_id}")
        return response

    except FileNotFoundError as e:
        logger.warning(f"Judgment list file not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error creating feature extraction job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create feature extraction job: {str(e)}",
        )


@router.get(
    "/featureset/extract-features/status/{job_id}",
    response_model=FeatureExtractionJobStatusResponse,
    summary="Check feature extraction job status",
    description="Check the status of a feature extraction job by its ID.",
)
async def get_feature_extraction_job_status(
    job_id: UUID,
    service: FeatureSetService = Depends(get_featureset_service),
) -> FeatureExtractionJobStatusResponse:
    """
    Get the status of a feature extraction job.

    Args:
        job_id: The UUID of the job
        service: Featureset service dependency

    Returns:
        Job status response with current status and details

    Raises:
        HTTPException: If job is not found
    """
    try:
        logger.debug(f"Checking status for feature extraction job {job_id}")

        job_status = await service.get_feature_extraction_job_status(job_id)

        if not job_status:
            logger.warning(f"Feature extraction job {job_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature extraction job with ID {job_id} not found",
            )

        logger.debug(f"Feature extraction job {job_id} status: {job_status.status}")
        return job_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving feature extraction job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status. Please try again later.",
        )


@router.post(
    "/training-data/merge",
    response_model=TrainingDataMergeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Merge featureset with judgment list to create training data",
    description="Merge a featureset CSV file with a judgment list CSV file to create training data.",
)
async def merge_training_data(
    request: TrainingDataMergeRequest = Body(
        ..., description="Training data merge configuration"
    ),
    service: FeatureSetService = Depends(get_featureset_service),
) -> TrainingDataMergeResponse:
    """
    Merge featureset file with judgment list file to create training data.

    Args:
        request: Request body containing judgment list and featureset filenames
        service: Featureset service dependency

    Returns:
        Response with training data creation status and file information

    Raises:
        HTTPException: If files not found or merge operation fails
    """
    try:
        logger.info(
            f"Received request to merge featureset '{request.featureset_filename}' "
            f"with judgment list '{request.judgment_list_filename}'"
        )

        response = await service.merge_featureset_with_judgment_list(request)

        logger.info(
            f"Successfully merged training data: {response.training_data_filename} "
            f"with {response.merged_records}/{response.total_records} records"
        )
        return response

    except FileNotFoundError as e:
        logger.warning(f"File not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error merging training data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge training data: {str(e)}",
        )
