"""
FastAPI router for XGBoost training endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.train.dependencies import get_training_service
from src.train.model import (
    ModelUploadRequest,
    ModelUploadResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingStatusResponse,
)
from src.train.service import TrainingService

# Create router
router = APIRouter(prefix="/train", tags=["training"])


@router.post(
    "/start",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start XGBoost training",
    description="Start training an XGBoost ranking model using XGBRanker in the background",
)
async def start_training(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_training_service),
):
    """
    Start XGBoost training with the provided parameters.

    Args:
        request: Training request parameters
        training_service: Training service instance

    Returns:
        Training response with job ID and status

    Raises:
        HTTPException: If training file not found or invalid parameters
    """
    try:
        logger.info(f"Starting training request: {request.training_file_name}")

        response = await training_service.start_training(request)

        logger.info(f"Training started successfully: {response.training_id}")
        return response

    except FileNotFoundError as e:
        logger.error(f"Training file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training file not found: {request.training_file_name}",
        )
    except ValueError as e:
        logger.error(f"Invalid training data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training data: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start training",
        )


@router.get(
    "/status/{training_id}",
    response_model=TrainingStatusResponse,
    summary="Get training status",
    description="Get the current status of a training job",
)
async def get_training_status(
    training_id: str, training_service: TrainingService = Depends(get_training_service)
) -> TrainingStatusResponse:
    """
    Get the status of a training job.

    Args:
        training_id: Training job ID
        training_service: Training service instance

    Returns:
        Training status response

    Raises:
        HTTPException: If training job not found
    """
    try:
        status_response = await training_service.get_training_status(training_id)

        if not status_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job not found: {training_id}",
            )

        return status_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get training status",
        )


@router.get(
    "/files",
    response_model=List[str],
    summary="Get available training files",
    description="Get list of available training data files",
)
async def get_available_training_files(
    training_service: TrainingService = Depends(get_training_service),
) -> List[str]:
    """
    Get list of available training data files.

    Args:
        training_service: Training service instance

    Returns:
        List of training file names
    """
    try:
        files = await training_service.get_available_training_files()
        return files

    except Exception as e:
        logger.error(f"Failed to get training files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get training files",
        )


@router.get(
    "/models",
    response_model=List[dict],
    summary="Get available trained models",
    description="Get list of available trained models",
)
async def get_available_models(
    training_service: TrainingService = Depends(get_training_service),
) -> List[dict]:
    """
    Get list of available trained models.

    Args:
        training_service: Training service instance

    Returns:
        List of model information dictionaries
    """
    try:
        models = await training_service.get_available_models()
        return models

    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get models",
        )


@router.post(
    "/cleanup",
    status_code=status.HTTP_200_OK,
    summary="Cleanup completed tasks",
    description="Clean up completed background training tasks",
)
async def cleanup_completed_tasks(
    training_service: TrainingService = Depends(get_training_service),
) -> dict:
    """
    Clean up completed background training tasks.

    Args:
        training_service: Training service instance

    Returns:
        Success message
    """
    try:
        await training_service.cleanup_completed_tasks()
        return {"message": "Completed tasks cleaned up successfully"}

    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup tasks",
        )


@router.post(
    "/upload-to-elasticsearch",
    response_model=ModelUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload model to Elasticsearch LTR",
    description="Upload a trained XGBoost model to Elasticsearch Learning to Rank plugin",
)
async def upload_model_to_elasticsearch(
    request: ModelUploadRequest,
    training_service: TrainingService = Depends(get_training_service),
) -> ModelUploadResponse:
    """
    Upload a trained XGBoost model to Elasticsearch Learning to Rank plugin.

    Args:
        request: Model upload request with model name and featureset name
        training_service: Training service instance

    Returns:
        Model upload response with success status and details

    Raises:
        HTTPException: If model not found or upload fails
    """
    try:
        logger.info(
            f"Starting model upload request: {request.model_name} to {request.featureset_name}"
        )

        response = await training_service.upload_model_to_elasticsearch(request)

        if response.success:
            logger.info(f"Model upload completed successfully: {response.model_name}")
        else:
            logger.error(f"Model upload failed: {response.message}")

        return response

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {request.model_name}",
        )
    except ValueError as e:
        logger.error(f"Invalid model file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model file: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload model to Elasticsearch",
        )
