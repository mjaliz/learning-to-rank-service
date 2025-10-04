"""
Service for training XGBoost ranking models using XGBRanker.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from loguru import logger
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker

from src.config import settings
from src.train.model import (
    ModelUploadRequest,
    ModelUploadResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    TrainingStatusResponse,
)
from src.train.repository import TrainingRepository


class TrainingService:
    """Service for training XGBoost ranking models."""

    def __init__(self, repository: TrainingRepository):
        """
        Initialize the training service.

        Args:
            repository: Training repository instance
        """
        self.repository = repository
        self._background_tasks: Dict[str, asyncio.Task] = {}

    async def start_training(self, request: TrainingRequest) -> TrainingResponse:
        """
        Start XGBoost training in the background.

        Args:
            request: Training request parameters

        Returns:
            Training response with job ID and status

        Raises:
            FileNotFoundError: If training file doesn't exist
            ValueError: If training data is invalid
        """
        # Generate unique training ID
        training_id = str(uuid.uuid4())

        # Create training job
        training_params = {
            "max_depth": request.max_depth,
            "learning_rate": request.learning_rate,
            "n_estimators": request.n_estimators,
            "random_state": request.random_state,
        }

        model_name = (
            request.model_name
            or f"xgboost_ranker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        job = self.repository.create_training_job(
            training_id=training_id,
            training_file_name=request.training_file_name,
            model_name=model_name,
            training_params=training_params,
        )

        # Start background training task
        task = asyncio.create_task(self._train_model_background(training_id, request))
        self._background_tasks[training_id] = task

        logger.info(
            f"Started training job {training_id} for file {request.training_file_name}"
        )

        return TrainingResponse(
            training_id=training_id,
            status=TrainingStatus.PENDING,
            message="Training job started successfully",
            started_at=job.started_at,
        )

    async def get_training_status(
        self, training_id: str
    ) -> Optional[TrainingStatusResponse]:
        """
        Get the status of a training job.

        Args:
            training_id: Training job ID

        Returns:
            Training status response or None if job not found
        """
        job = self.repository.get_training_job(training_id)
        if not job:
            return None

        # Check if background task is still running
        if training_id in self._background_tasks:
            task = self._background_tasks[training_id]
            if task.done():
                # Task completed, remove from active tasks
                del self._background_tasks[training_id]

                # Check if task raised an exception
                try:
                    task.result()
                except Exception as e:
                    logger.error(f"Training task {training_id} failed: {e}")
                    self.repository.update_training_job(
                        training_id,
                        status=TrainingStatus.FAILED,
                        error_message=str(e),
                        completed_at=datetime.now(),
                    )

        return TrainingStatusResponse(
            training_id=job.training_id,
            status=job.status,
            progress=job.progress,
            message=self._get_status_message(job.status),
            started_at=job.started_at,
            completed_at=job.completed_at,
            model_path=job.model_path,
            training_metrics=job.training_metrics,
            error_message=job.error_message,
        )

    async def _train_model_background(
        self, training_id: str, request: TrainingRequest
    ) -> None:
        """
        Train the XGBoost model in the background.

        Args:
            training_id: Training job ID
            request: Training request parameters
        """
        try:
            # Update status to running
            self.repository.update_training_job(
                training_id, status=TrainingStatus.RUNNING, progress=0.0
            )

            # Load training data
            logger.info(f"Loading training data for job {training_id}")
            df = await self.repository.get_training_data(request.training_file_name)

            # Prepare data for XGBRanker
            logger.info(f"Preparing data for XGBRanker for job {training_id}")
            X, y, groups, query_ids = self._prepare_ranking_data(df)

            # Split data for validation using GroupShuffleSplit to preserve query groups
            gss = GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=request.random_state
            )
            train_idx, val_idx = next(gss.split(X, y, groups=query_ids))

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            query_ids_train, query_ids_val = query_ids[train_idx], query_ids[val_idx]

            # Recalculate group sizes for training and validation sets
            _, groups_train = np.unique(query_ids_train, return_counts=True)
            _, groups_val = np.unique(query_ids_val, return_counts=True)

            # Create XGBRanker model
            model = XGBRanker(
                # max_depth=request.max_depth,
                # learning_rate=request.learning_rate,
                # n_estimators=request.n_estimators,
                # random_state=request.random_state,
                tree_method="hist",
                device="cpu",
                lambdarank_pair_method="topk",
                lambdarank_num_pair_per_sample=8,
                objective="rank:ndcg",
                eval_metric=["ndcg@1", "ndcg@10", "ndcg@20"],
                early_stopping_rounds=20,
            )

            # Update progress
            self.repository.update_training_job(training_id, progress=20.0)

            # Train the model
            logger.info(f"Training XGBRanker model for job {training_id}")
            model.fit(
                X_train,
                y_train,
                group=groups_train,
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                verbose=True,
            )

            # Update progress
            self.repository.update_training_job(training_id, progress=80.0)

            # Calculate training metrics
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)

            # Calculate NDCG scores
            train_ndcg = self._calculate_ndcg_with_sklearn(
                y_train, train_predictions, query_ids_train
            )
            val_ndcg = self._calculate_ndcg_with_sklearn(
                y_val, val_predictions, query_ids_val
            )

            training_metrics = {
                "train_ndcg": train_ndcg,
                "val_ndcg": val_ndcg,
                "n_features": X.shape[1],
                "n_samples": len(X),
                "n_groups": len(np.unique(query_ids)),
            }

            # Save the model
            model_name = (
                request.model_name
                or f"xgboost_ranker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            model_path = await self.repository.save_model(
                model, model_name, training_id, request.feature_map_file_name
            )

            # Update job as completed
            self.repository.update_training_job(
                training_id,
                status=TrainingStatus.COMPLETED,
                progress=100.0,
                completed_at=datetime.now(),
                model_path=model_path,
                training_metrics=training_metrics,
            )

            logger.info(f"Training completed successfully for job {training_id}")

        except Exception as e:
            logger.error(f"Training failed for job {training_id}: {e}")
            self.repository.update_training_job(
                training_id,
                status=TrainingStatus.FAILED,
                error_message=str(e),
                completed_at=datetime.now(),
            )
            raise

    def _prepare_ranking_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for XGBRanker training.

        Args:
            df: Training data DataFrame

        Returns:
            Tuple of (features, labels, groups_array, query_ids)
        """
        # Separate features and labels
        feature_columns = [
            col for col in df.columns if col not in ["qid", "relevance_grade"]
        ]
        X = df[feature_columns].values
        y = df["relevance_grade"].values

        # Create groups based on query IDs
        query_ids = df["qid"].values

        # Sort by query ID to ensure proper grouping
        sort_indices = np.argsort(query_ids)
        X = X[sort_indices]
        y = y[sort_indices]
        query_ids = query_ids[sort_indices]

        # Create group sizes for XGBRanker
        # XGBRanker expects group parameter to be an array of group sizes
        # in the order that groups appear in the sorted data
        unique_groups, group_counts = np.unique(query_ids, return_counts=True)
        groups_array = group_counts

        logger.info(
            f"Prepared data: {len(X)} samples, {len(feature_columns)} features, {len(unique_groups)} groups"
        )

        return X, y, groups_array, query_ids

    def _calculate_ndcg_with_sklearn(
        self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) score using sklearn.

        Args:
            y_true: True relevance scores
            y_pred: Predicted scores
            groups: Group identifiers
            k: Number of top results to consider

        Returns:
            Average NDCG score across all groups
        """
        ndcg_scores = []

        # Group data by query
        unique_groups = np.unique(groups)

        for group_id in unique_groups:
            # Get indices for this group
            group_mask = groups == group_id
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]

            if len(group_y_true) == 0:
                continue

            # Skip queries with only 1 document as NDCG is not meaningful
            if len(group_y_true) == 1:
                # For single document queries, NDCG is always 1.0 if the document is relevant
                # or 0.0 if not relevant (assuming relevance > 0 means relevant)
                ndcg = 1.0 if group_y_true[0] > 0 else 0.0
                ndcg_scores.append(ndcg)
                continue

            # sklearn ndcg_score expects 2D arrays where each row is a query
            # and each column is a document. For a single query, we reshape to (1, n_docs)
            y_true_2d = group_y_true.reshape(1, -1)
            y_pred_2d = group_y_pred.reshape(1, -1)

            # Calculate NDCG using sklearn
            ndcg = ndcg_score(y_true_2d, y_pred_2d, k=k)
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def _get_status_message(self, status: TrainingStatus) -> str:
        """
        Get human-readable status message.

        Args:
            status: Training status

        Returns:
            Status message
        """
        messages = {
            TrainingStatus.PENDING: "Training job is pending",
            TrainingStatus.RUNNING: "Training is in progress",
            TrainingStatus.COMPLETED: "Training completed successfully",
            TrainingStatus.FAILED: "Training failed",
        }
        return messages.get(status, "Unknown status")

    async def get_available_training_files(self) -> list:
        """
        Get list of available training files.

        Returns:
            List of training file names
        """
        return await self.repository.get_available_training_files()

    async def get_available_models(self) -> list:
        """
        Get list of available trained models.

        Returns:
            List of model information
        """
        return await self.repository.get_available_models()

    async def cleanup_completed_tasks(self) -> None:
        """Clean up completed background tasks."""
        completed_tasks = []

        for training_id, task in self._background_tasks.items():
            if task.done():
                completed_tasks.append(training_id)

        for training_id in completed_tasks:
            del self._background_tasks[training_id]

        if completed_tasks:
            logger.info(f"Cleaned up {len(completed_tasks)} completed training tasks")

    async def upload_model_to_elasticsearch(
        self, request: ModelUploadRequest
    ) -> ModelUploadResponse:
        """
        Upload a trained XGBoost model to Elasticsearch Learning to Rank plugin.

        Args:
            request: Model upload request with model name and featureset name

        Returns:
            Model upload response with success status and details

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        try:
            logger.info(
                f"Starting model upload: {request.model_name} to featureset {request.featureset_name}"
            )

            # Find the model file
            model_path = await self._find_model_file(request.model_name)
            if not model_path:
                raise FileNotFoundError(f"Model file not found: {request.model_name}")

            # Load the model dump
            logger.info(f"Loading model dump from {model_path}")
            model_definition = await self._load_model_dump(model_path)

            # Create the payload
            payload = self._create_ltr_model_payload(
                model_definition, request.model_name
            )

            # Upload to Elasticsearch
            success = await self._upload_to_elasticsearch(
                payload, request.featureset_name
            )

            if success:
                logger.info(
                    f"Model {request.model_name} uploaded successfully to featureset {request.featureset_name}"
                )
                return ModelUploadResponse(
                    success=True,
                    message=f"Model '{request.model_name}' uploaded successfully to featureset '{request.featureset_name}'",
                    model_name=request.model_name,
                    featureset_name=request.featureset_name,
                    uploaded_at=datetime.now(),
                )
            else:
                logger.error(f"Failed to upload model {request.model_name}")
                return ModelUploadResponse(
                    success=False,
                    message=f"Failed to upload model '{request.model_name}' to featureset '{request.featureset_name}'",
                    model_name=request.model_name,
                    featureset_name=request.featureset_name,
                    uploaded_at=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error uploading model {request.model_name}: {e}")
            return ModelUploadResponse(
                success=False,
                message=f"Error uploading model: {str(e)}",
                model_name=request.model_name,
                featureset_name=request.featureset_name,
                uploaded_at=datetime.now(),
            )

    async def _find_model_file(self, model_name: str) -> Optional[str]:
        """
        Find the model file by name in the outputs/models directory.

        Args:
            model_name: Name of the model to find

        Returns:
            Path to the model file or None if not found
        """
        models_dir = Path(settings.OUTPUT_DIR) / "models"

        # Look for JSON files that contain the model name
        for model_file in models_dir.glob("*.json"):
            if model_name in model_file.name:
                return str(model_file)

        return None

    async def _load_model_dump(self, file_path: str) -> list:
        """
        Load the XGBoost model from JSON file.

        Args:
            file_path: Path to the XGBoost model JSON file

        Returns:
            Model definition as a list of tree objects

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        try:
            with open(file_path, "r") as f:
                # Read the entire file as string - this is the binary JSON format
                model_json_str = f.read()

            # Parse and return as Python object (list of tree objects)
            return json.loads(model_json_str)

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def _create_ltr_model_payload(
        self, model_definition: list, model_name: str
    ) -> Dict[str, Any]:
        """
        Create the payload for uploading the model to Elasticsearch LTR.

        Args:
            model_definition: The XGBoost model definition as list of trees
            model_name: Name for the model in Elasticsearch

        Returns:
            Dictionary containing the model payload
        """
        return {
            "model": {
                "name": model_name,
                "model": {
                    "type": "model/xgboost+json",
                    "definition": model_definition,
                },
            }
        }

    def _get_elasticsearch_client(self) -> Elasticsearch:
        """
        Create and return Elasticsearch client using configuration from settings.

        Returns:
            Configured Elasticsearch client
        """
        return Elasticsearch(
            hosts=[settings.ELASTIC_HOST],
            timeout=300,
            basic_auth=(settings.ELASTIC_USER, settings.ELASTIC_PASSWORD),
            http_auth=(settings.ELASTIC_USER, settings.ELASTIC_PASSWORD),
        )

    async def _upload_to_elasticsearch(
        self, payload: Dict[str, Any], featureset_name: str
    ) -> bool:
        """
        Upload the model to Elasticsearch Learning to Rank plugin.

        Args:
            payload: The model payload to upload
            featureset_name: Name of the featureset to associate with the model

        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Get Elasticsearch client
            es = self._get_elasticsearch_client()

            logger.info(f"Uploading model to featureset '{featureset_name}'...")

            # Use Elasticsearch client's transport to make the LTR API call
            response = es.transport.perform_request(
                method="POST",
                url=f"/_ltr/_featureset/{featureset_name}/_createmodel",
                body=json.dumps(payload),
            )

            logger.info("Model uploaded successfully!")
            logger.debug(f"Response: {response}")
            return True

        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False
