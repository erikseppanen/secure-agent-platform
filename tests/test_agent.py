from collections.abc import Sequence

import pytest

from app.agent import run_agent
from app.llm.base import LLMProvider
from app.llm.models import Message, ModelResponse, ToolCall, ToolDefinition


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.messages: list[tuple[Message, ...]] = []

    async def run(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        self.messages.append(tuple(messages))
        return next(self._responses)


@pytest.mark.asyncio
async def test_agent_returns_provider_text() -> None:
    provider = FakeProvider([ModelResponse(text="hello")])

    assert await run_agent("hi", provider=provider) == "hello"


@pytest.mark.asyncio
async def test_agent_executes_normalized_tool_call() -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        id="tool-1",
                        name="get_system_status",
                        arguments={"service": "billing"},
                    ),
                ),
                stop_reason="tool_use",
            ),
            ModelResponse(text="Billing is healthy."),
        ]
    )

    assert await run_agent("How is billing?", provider=provider) == (
        "Billing is healthy."
    )
    tool_result = provider.messages[1][-1].tool_results[0]
    assert tool_result.tool_call_id == "tool-1"
    assert '"status": "healthy"' in tool_result.content
