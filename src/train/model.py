"""
Pydantic models for XGBoost training functionality.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrainingStatus(str, Enum):
    """Training status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingRequest(BaseModel):
    """Request model for starting XGBoost training."""

    training_file_name: str = Field(
        ...,
        description="Name of the training data file (CSV format)",
        example="20251001_171911_training_data.csv",
    )
    feature_map_file_name: str = Field(
        ...,
        description="Name of the feature map file (fmap.txt format)",
        example="20251001_171911_fmap.txt",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Custom name for the trained model. If not provided, will be auto-generated",
        example="product_ranking_model_v1",
    )
    max_depth: int = Field(
        default=6, description="Maximum depth of the tree", ge=1, le=20
    )
    learning_rate: float = Field(
        default=0.1, description="Learning rate for the model", ge=0.01, le=1.0
    )
    n_estimators: int = Field(
        default=100, description="Number of boosting rounds", ge=10, le=1000
    )
    random_state: int = Field(
        default=42, description="Random state for reproducibility"
    )


class TrainingResponse(BaseModel):
    """Response model for training request."""

    training_id: str = Field(..., description="Unique identifier for the training job")
    status: TrainingStatus = Field(
        ..., description="Current status of the training job"
    )
    message: str = Field(
        ..., description="Human-readable message about the training status"
    )
    started_at: datetime = Field(..., description="Timestamp when training was started")


class TrainingStatusResponse(BaseModel):
    """Response model for training status check."""

    training_id: str = Field(..., description="Unique identifier for the training job")
    status: TrainingStatus = Field(
        ..., description="Current status of the training job"
    )
    progress: Optional[float] = Field(
        default=None,
        description="Training progress percentage (0-100)",
        ge=0.0,
        le=100.0,
    )
    message: str = Field(
        ..., description="Human-readable message about the training status"
    )
    started_at: datetime = Field(..., description="Timestamp when training was started")
    completed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when training was completed"
    )
    model_path: Optional[str] = Field(
        default=None, description="Path to the saved model file (if training completed)"
    )
    training_metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Training metrics (if available)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if training failed"
    )


class ModelUploadRequest(BaseModel):
    """Request model for uploading a trained model to Elasticsearch LTR."""

    model_name: str = Field(
        ...,
        description="Name of the trained model to upload",
        example="product_ranking_model_v1_20251004_114800_88768340-cf90-4af0-b388-3bf9f9fd9ebd",
    )
    featureset_name: str = Field(
        ...,
        description="Name of the featureset to associate with the model",
        example="first_featureset",
    )


class ModelUploadResponse(BaseModel):
    """Response model for model upload request."""

    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(
        ..., description="Human-readable message about the upload result"
    )
    model_name: str = Field(..., description="Name of the uploaded model")
    featureset_name: str = Field(..., description="Name of the featureset")
    uploaded_at: datetime = Field(
        ..., description="Timestamp when the model was uploaded"
    )


class TrainingJob(BaseModel):
    """Internal model for tracking training jobs."""

    training_id: str
    status: TrainingStatus
    training_file_name: str
    model_name: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    model_path: Optional[str] = None
    training_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    training_params: Dict[str, any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)
