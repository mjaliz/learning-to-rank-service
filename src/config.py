"""
Configuration module for loading environment variables using pydantic-settings.
All configuration fields must be UPPERCASE and are case-sensitive.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All field names must be UPPERCASE in the .env file.
    Extra fields in .env file will be ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # PostgreSQL searchworker settings
    POSTGRES_SEARCHWORKER_HOST: str = Field(
        default="localhost", description="PostgreSQL host"
    )
    POSTGRES_SEARCHWORKER_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_SEARCHWORKER_DB: str = Field(
        default="", description="PostgreSQL database name"
    )
    POSTGRES_SEARCHWORKER_USER: str = Field(default="", description="PostgreSQL user")
    POSTGRES_SEARCHWORKER_PASSWORD: str = Field(
        default="", description="PostgreSQL password"
    )

    # PostgreSQL settings
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field(default="", description="PostgreSQL database name")
    POSTGRES_USER: str = Field(default="", description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(default="", description="PostgreSQL password")

    # Elasticsearch settings
    ELASTIC_HOST: str = Field(
        default="http://localhost", description="Elasticsearch host URL"
    )
    ELASTIC_PORT: int = Field(default=9200, description="Elasticsearch port")
    ELASTIC_INDEX: str = Field(default="", description="Elasticsearch index name")
    ELASTIC_USER: str = Field(default="", description="Elasticsearch username")
    ELASTIC_PASSWORD: str = Field(default="", description="Elasticsearch password")

    # File storage settings
    UPLOAD_DIR: str = Field(
        default="uploads", description="Directory for uploaded SQL files"
    )
    OUTPUT_DIR: str = Field(
        default="outputs", description="Directory for generated judgment list files"
    )
    MAX_FILE_SIZE: int = Field(
        default=100 * 1024 * 1024, description="Maximum file size in bytes (100MB)"
    )

    # API settings
    API_HOST: str = Field(default="0.0.0.0", description="API server host")
    API_PORT: int = Field(default=8000, description="API server port")


# Singleton instance
settings = Settings()
