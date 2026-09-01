from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: Literal["anthropic_api", "claude_agent_sdk"] = "anthropic_api"

    anthropic_api_key: str | None = None
    anthropic_workspace_id: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 1024

    claude_agent_model: str = "claude-sonnet-4-6"
    claude_agent_max_turns: int = 8
    claude_agent_use_subscription: bool = True

    agent_max_steps: int = 8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
