"""Application configuration via environment variables.

Uses pydantic-settings to load and validate all configuration
from .env file. Never hardcode secrets.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for TalentRoute AI.

    All values are loaded from environment variables or .env file.
    Pydantic validates types and provides defaults where appropriate.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM Provider ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.1

    # --- Embedding Model ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Paths ---
    faiss_index_path: str = "app/data/vectorstore/catalog.index"
    faiss_metadata_path: str = "app/data/vectorstore/catalog_metadata.pkl"
    catalog_path: str = "app/data/raw/SHL-Catalogue.txt"

    # --- Retrieval ---
    retrieval_top_k: int = 30
    max_recommendations: int = 10

    # --- Logging ---
    log_level: str = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def absolute_faiss_index_path(self) -> Path:
        """Return absolute path to FAISS index."""
        return self.project_root / self.faiss_index_path

    @property
    def absolute_faiss_metadata_path(self) -> Path:
        """Return absolute path to FAISS metadata."""
        return self.project_root / self.faiss_metadata_path

    @property
    def absolute_catalog_path(self) -> Path:
        """Return absolute path to catalog JSON."""
        return self.project_root / self.catalog_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton.

    Returns:
        Settings: Validated application configuration.
    """
    return Settings()
