from dataclasses import dataclass
from typing import Any

import pytest

from app.llm.claude_agent_sdk import ClaudeAgentSdkProvider
from app.llm.models import Message, ToolDefinition


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AssistantMessage:
    content: list[Any]


@dataclass
class ResultMessage:
    result: str | None
    is_error: bool = False
    errors: list[str] | None = None


class FakeSdk:
    TextBlock = TextBlock
    ToolUseBlock = ToolUseBlock
    AssistantMessage = AssistantMessage
    ResultMessage = ResultMessage

    def __init__(self) -> None:
        self.options = None
        self.sdk_tools = []

    def tool(self, name, description, input_schema):
        def decorate(handler):
            handler.tool_name = name
            return handler

        return decorate

    def create_sdk_mcp_server(self, *, name, version, tools):
        self.sdk_tools = tools
        return {"name": name, "version": version, "tools": tools}

    def ClaudeAgentOptions(self, **kwargs):
        self.options = kwargs
        return kwargs

    async def query(self, *, prompt, options):
        yield AssistantMessage(
            [ToolUseBlock("call-1", "mcp__secure_agent__status", {"x": 1})]
        )
        yield ResultMessage("done")


@pytest.mark.asyncio
async def test_sdk_provider_uses_subscription_auth_and_app_tools_only() -> None:
    sdk = FakeSdk()
    provider = ClaudeAgentSdkProvider(
        model="sonnet",
        max_turns=4,
        system_prompt="test",
        sdk=sdk,
    )
    definition = ToolDefinition(
        name="status",
        description="status",
        input_schema={"type": "object"},
        handler=lambda arguments: {"ok": arguments["x"]},
    )

    response = await provider.run([Message(role="user", text="go")], [definition])

    assert response.text == "done"
    assert response.tool_calls[0].name == "status"
    assert sdk.options["env"]["ANTHROPIC_API_KEY"] == ""
    assert sdk.options["tools"] == []
    assert sdk.options["strict_mcp_config"] is True
    assert sdk.options["allowed_tools"] == ["mcp__secure_agent__status"]

    tool_result = await sdk.sdk_tools[0]({"x": 2})
    assert '"ok": 2' in tool_result["content"][0]["text"]
