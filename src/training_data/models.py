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
