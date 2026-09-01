from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.llm.models import Message, ModelResponse, ToolDefinition


class LLMProvider(ABC):
    @abstractmethod
    async def run(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """Run one provider turn and return a provider-neutral response."""
