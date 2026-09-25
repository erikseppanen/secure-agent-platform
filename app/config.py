from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_workspace_id: str
    anthropic_model: str = "claude-sonnet-4-6"
    database_url: str = "postgresql://sap:sap@localhost:5432/sap"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    log_level: str = "INFO"
    log_file: str = "logs/app.jsonl"
    log_viewer_enabled: bool = True
    trace_payloads: bool = True
    trace_max_chars: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
