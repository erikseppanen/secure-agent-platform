from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_workspace_id: str
    anthropic_model: str = "claude-sonnet-4-6"
    database_url: str = "postgresql://sap:sap@localhost:5432/sap"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
