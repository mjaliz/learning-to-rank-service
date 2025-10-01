"""
Data models for training data processing jobs.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlmodel import Column, DateTime, SQLModel
from sqlmodel import Field as SQLField


class JobStatus(str, Enum):
    """Status of a judgment list file creation job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JudgmentListJob(SQLModel, table=True):
    """Database model for judgment list creation jobs."""

    __tablename__ = "judgment_list_jobs"

    id: UUID = SQLField(
        default_factory=uuid4,
        primary_key=True,
        nullable=False,
    )
    filename: str = SQLField(nullable=False, index=True)
    status: JobStatus = SQLField(default=JobStatus.PENDING, nullable=False, index=True)
    sql_file_path: str = SQLField(nullable=False)
    output_file_path: Optional[str] = SQLField(default=None)
    error_message: Optional[str] = SQLField(default=None)
    created_at: datetime = SQLField(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        default_factory=datetime.utcnow,
    )
    started_at: Optional[datetime] = SQLField(
        sa_column=Column(DateTime(timezone=True), nullable=True), default=None
    )
    completed_at: Optional[datetime] = SQLField(
        sa_column=Column(DateTime(timezone=True), nullable=True), default=None
    )


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""

    job_id: UUID
    filename: str
    status: JobStatus
    output_file_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobCreateResponse(BaseModel):
    """Response model for job creation."""

    job_id: UUID
    filename: str
    status: JobStatus
    message: str = Field(
        default="Job created successfully. Processing will begin shortly."
    )


class FeatureTemplate(BaseModel):
    """Template for a feature in Elasticsearch."""

    lang: str = Field(default="painless", description="Script language")
    source: str = Field(description="Script source code")


class Feature(BaseModel):
    """A single feature definition for Elasticsearch featureset."""

    name: str = Field(description="Name of the feature")
    params: list = Field(default_factory=list, description="Feature parameters")
    template_language: str = Field(
        default="script_feature", description="Template language type"
    )
    template: FeatureTemplate = Field(description="Feature template configuration")


class FeatureSetRequest(BaseModel):
    """Request model for creating a featureset."""

    features: list[Feature] = Field(
        min_length=1, description="List of features for the featureset"
    )


class FeatureSetResponse(BaseModel):
    """Response model for featureset creation."""

    featureset_name: str = Field(description="Name of the created featureset")
    features_count: int = Field(description="Number of features in the featureset")
    message: str = Field(default="Featureset created successfully")
    acknowledged: bool = Field(
        default=True, description="Elasticsearch acknowledgment status"
    )


class FeatureExtractionRequest(BaseModel):
    """Request model for extracting features from judgment list."""

    judgment_list_filename: str = Field(
        description="Name of the judgment list CSV file"
    )
    featureset_name: str = Field(description="Name of the featureset to use")


class ProductFeatures(BaseModel):
    """Features extracted for a single product."""

    product_id: str = Field(description="Product ID")
    features: list[dict] = Field(
        default_factory=list, description="List of feature values"
    )


class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction."""

    featureset_name: str = Field(description="Name of the featureset used")
    total_products: int = Field(description="Total number of products processed")
    products_with_features: int = Field(
        description="Number of products with extracted features"
    )
    product_features: list[ProductFeatures] = Field(
        default_factory=list, description="Feature vectors for each product"
    )
