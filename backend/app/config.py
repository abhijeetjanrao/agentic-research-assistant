"""
Centralized application configuration.

Why this file exists:
    Every other module (DB, agents, RAG, API routes) needs settings like
    API keys, model names, or directory paths. Instead of each module
    calling os.getenv() independently (which silently returns None on typos
    and has no type validation), we define one Settings object, validated
    once at startup by pydantic. If a required variable is missing or the
    wrong type, the app fails fast with a clear error instead of failing
    mysteriously three layers deep inside an agent call.

Usage:
    from app.config import get_settings
    settings = get_settings()
    settings.gemini_model
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App metadata ---
    app_name: str = "Agentic Research Assistant"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # --- Gemini ---
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.3

    # --- Web search ---
    tavily_api_key: str

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- FAISS ---
    faiss_index_dir: str = "./data/faiss_index"

    # --- MySQL ---
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str
    mysql_password: str
    mysql_database: str

    # --- Uploads ---
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 25

    # --- API ---
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:8501"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def mysql_url(self) -> str:
        """SQLAlchemy connection string, built from discrete parts.

        We keep host/user/password/db separate in .env (easier to override
        individually in CI/CD or docker-compose) and only assemble the full
        URL here, in one place.
        """
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor.

    lru_cache ensures the .env file is parsed and validated only once per
    process, and every module that calls get_settings() shares the same
    Settings instance instead of re-reading disk each time.
    """
    return Settings()
