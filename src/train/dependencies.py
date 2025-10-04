"""
Dependencies for the training module.
"""

from fastapi import Depends

from src.train.repository import TrainingRepository
from src.train.service import TrainingService


def get_training_repository() -> TrainingRepository:
    """
    Get training repository instance.

    Returns:
        TrainingRepository instance
    """
    return TrainingRepository()


def get_training_service(
    repository: TrainingRepository = Depends(get_training_repository),
) -> TrainingService:
    """
    Get training service instance.

    Args:
        repository: Training repository instance

    Returns:
        TrainingService instance
    """
    return TrainingService(repository)
