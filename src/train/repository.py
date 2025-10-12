"""
Repository for managing training data and model storage.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config import settings
from src.train.model import TrainingJob, TrainingStatus


class TrainingRepository:
    """Repository for managing training data and model storage."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(TrainingRepository, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the training repository."""
        if not self._initialized:
            self.output_dir = Path(settings.OUTPUT_DIR)
            self.models_dir = self.output_dir / "models"
            self.training_jobs: Dict[str, TrainingJob] = {}

            # Ensure directories exist
            self.models_dir.mkdir(parents=True, exist_ok=True)

            TrainingRepository._initialized = True

    async def get_training_data(self, file_name: str) -> pd.DataFrame:
        """
        Load training data from CSV file.

        Args:
            file_name: Name of the training data file

        Returns:
            DataFrame containing the training data

        Raises:
            FileNotFoundError: If the training file doesn't exist
            ValueError: If the training data format is invalid
        """
        file_path = self.output_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_name}")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Validate required columns
            required_columns = ["qid", "relevance_grade"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Check if we have feature columns (columns other than qid and relevance_grade)
            feature_columns = [
                col for col in df.columns if col not in ["qid", "relevance_grade"]
            ]
            if not feature_columns:
                raise ValueError("No feature columns found in training data")

            logger.info(
                f"Loaded training data: {len(df)} rows, {len(feature_columns)} features"
            )
            return df

        except Exception as e:
            logger.error(f"Error loading training data from {file_name}: {e}")
            raise ValueError(f"Invalid training data format: {e}")

    async def save_model(
        self, model, model_name: str, training_id: str, feature_map_file_name: str
    ) -> str:
        """
        Save the trained model to disk using XGBoost model dump format.

        Args:
            model: The trained XGBoost model
            model_name: Name for the model
            training_id: Training job ID
            feature_map_file_name: Name of the feature map file

        Returns:
            Path to the saved model file
        """
        # Generate model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.json"
        model_path = self.models_dir / model_filename

        # Get the feature map file path
        fmap_path = self.output_dir / feature_map_file_name

        if not fmap_path.exists():
            raise FileNotFoundError(
                f"Feature map file not found: {feature_map_file_name}"
            )

        # Get model dump using the feature map
        model_dump = model.get_booster().get_dump(
            fmap=str(fmap_path), dump_format="json"
        )

        # Parse the model dump
        parsed_model_dump = [json.loads(md) for md in model_dump]

        # Save the human-readable model dump
        with open(model_path, "w") as output:
            json.dump(parsed_model_dump, output, indent=2)

        logger.info(f"Model saved to: {model_path}")
        return str(model_path)

    async def load_model(self, model_path: str):
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the model file

        Returns:
            The loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from: {model_path}")
        return model

    def create_training_job(
        self,
        training_id: str,
        training_file_name: str,
        model_name: Optional[str],
        training_params: Dict,
    ) -> TrainingJob:
        """
        Create a new training job.

        Args:
            training_id: Unique identifier for the training job
            training_file_name: Name of the training data file
            model_name: Name for the model
            training_params: Training parameters

        Returns:
            The created training job
        """
        job = TrainingJob(
            training_id=training_id,
            status=TrainingStatus.PENDING,
            training_file_name=training_file_name,
            model_name=model_name,
            started_at=datetime.now(),
            training_params=training_params,
        )

        self.training_jobs[training_id] = job
        logger.info(f"Created training job: {training_id}")
        return job

    def get_training_job(self, training_id: str) -> Optional[TrainingJob]:
        """
        Get a training job by ID.

        Args:
            training_id: Training job ID

        Returns:
            The training job if found, None otherwise
        """
        return self.training_jobs.get(training_id)

    def update_training_job(self, training_id: str, **updates) -> bool:
        """
        Update a training job.

        Args:
            training_id: Training job ID
            **updates: Fields to update

        Returns:
            True if job was updated, False if job not found
        """
        if training_id not in self.training_jobs:
            return False

        job = self.training_jobs[training_id]
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)

        logger.info(f"Updated training job {training_id}: {list(updates.keys())}")
        return True

    def get_all_training_jobs(self) -> List[TrainingJob]:
        """
        Get all training jobs.

        Returns:
            List of all training jobs
        """
        return list(self.training_jobs.values())

    def delete_training_job(self, training_id: str) -> bool:
        """
        Delete a training job.

        Args:
            training_id: Training job ID

        Returns:
            True if job was deleted, False if job not found
        """
        if training_id in self.training_jobs:
            del self.training_jobs[training_id]
            logger.info(f"Deleted training job: {training_id}")
            return True
        return False

    async def get_available_training_files(self) -> List[str]:
        """
        Get list of available training data files.

        Returns:
            List of training file names
        """
        training_files = []

        for file_path in self.output_dir.glob("*.csv"):
            if "training_data" in file_path.name:
                training_files.append(file_path.name)

        return sorted(training_files)

    async def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get list of available trained models.

        Returns:
            List of model information dictionaries
        """
        models = []

        for model_path in self.models_dir.glob("*.json"):
            stat = model_path.stat()
            models.append(
                {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return sorted(models, key=lambda x: x["modified_at"], reverse=True)
