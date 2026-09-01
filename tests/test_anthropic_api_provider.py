from types import SimpleNamespace

import pytest

from app.llm.anthropic_api import AnthropicApiProvider
from app.llm.models import Message, ToolCall, ToolResult


class FakeMessages:
    def __init__(self) -> None:
        self.kwargs = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Checking."),
                SimpleNamespace(
                    type="tool_use",
                    id="tool-1",
                    name="status",
                    input={"service": "billing"},
                ),
            ],
            stop_reason="tool_use",
        )


@pytest.mark.asyncio
async def test_anthropic_provider_normalizes_response_and_history() -> None:
    messages_api = FakeMessages()
    client = SimpleNamespace(messages=messages_api)
    provider = AnthropicApiProvider(
        api_key="test",
        model="test-model",
        system_prompt="test",
        client=client,
    )
    history = [
        Message(role="user", text="status?"),
        Message(
            role="assistant",
            tool_calls=(ToolCall("old", "status", {"service": "billing"}),),
        ),
        Message(
            role="user",
            tool_results=(ToolResult("old", '{"status": "healthy"}'),),
        ),
    ]

    response = await provider.run(history, [])

    assert response.stop_reason == "tool_use"
    assert response.text == "Checking."
    assert response.tool_calls[0].name == "status"
    assert messages_api.kwargs["messages"][2]["content"][0]["type"] == ("tool_result")
