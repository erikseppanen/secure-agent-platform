from app.config import Settings
from app.llm.anthropic_api import AnthropicApiProvider
from app.llm.base import LLMProvider
from app.llm.claude_agent_sdk import ClaudeAgentSdkProvider

SYSTEM_PROMPT = (
    "You are an enterprise operations assistant. "
    "Use available tools when they are needed to answer factual questions "
    "about internal systems."
)


def create_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic_api":
        return AnthropicApiProvider(
            api_key=settings.anthropic_api_key or "",
            workspace_id=settings.anthropic_workspace_id,
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system_prompt=SYSTEM_PROMPT,
        )

    if settings.llm_provider == "claude_agent_sdk":
        return ClaudeAgentSdkProvider(
            model=settings.claude_agent_model,
            max_turns=settings.claude_agent_max_turns,
            system_prompt=SYSTEM_PROMPT,
            use_subscription=settings.claude_agent_use_subscription,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
