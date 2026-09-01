import inspect
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.llm.base import LLMProvider
from app.llm.models import Message, ModelResponse, ToolCall, ToolDefinition

_DISALLOWED_BUILTIN_TOOLS = [
    "Agent",
    "AskUserQuestion",
    "Bash",
    "Edit",
    "EnterPlanMode",
    "ExitPlanMode",
    "Glob",
    "Grep",
    "NotebookEdit",
    "Read",
    "Skill",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
]


class ClaudeAgentSdkProvider(LLMProvider):
    """Claude-plan provider backed by the Claude Agent SDK.

    This adapter intentionally clears inherited API credentials so the SDK
    uses the local Claude login and its Agent SDK credit allocation instead
    of silently switching to pay-per-token API billing.
    """

    def __init__(
        self,
        *,
        model: str,
        max_turns: int,
        system_prompt: str,
        use_subscription: bool = True,
        sdk: Any | None = None,
    ) -> None:
        if sdk is None:
            import claude_agent_sdk as sdk_module

            sdk = sdk_module

        self._sdk = sdk
        self._model = model
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._use_subscription = use_subscription

    async def run(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        sdk_tools = [self._to_sdk_tool(definition) for definition in tools]
        mcp_servers: dict[str, Any] = {}
        allowed_tools: list[str] = []

        if sdk_tools:
            mcp_servers["secure_agent"] = self._sdk.create_sdk_mcp_server(
                name="secure-agent-tools",
                version="1.0.0",
                tools=sdk_tools,
            )
            allowed_tools = [
                f"mcp__secure_agent__{definition.name}" for definition in tools
            ]

        env = {}
        if self._use_subscription:
            env = {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "",
            }

        options = self._sdk.ClaudeAgentOptions(
            tools=[],
            model=self._model,
            system_prompt=self._system_prompt,
            max_turns=self._max_turns,
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            disallowed_tools=_DISALLOWED_BUILTIN_TOOLS,
            strict_mcp_config=True,
            setting_sources=[],
            skills=[],
            env=env,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        final_text = ""

        async for message in self._query(self._render_prompt(messages), options):
            if isinstance(message, self._sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, self._sdk.TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, self._sdk.ToolUseBlock):
                        tool_calls.append(
                            ToolCall(
                                id=block.id,
                                name=self._strip_mcp_prefix(block.name),
                                arguments=block.input,
                            )
                        )
            elif isinstance(message, self._sdk.ResultMessage):
                if message.is_error:
                    details = "; ".join(message.errors) if message.errors else ""
                    raise RuntimeError(
                        details or message.result or "Claude Agent SDK failed"
                    )
                final_text = message.result or ""

        return ModelResponse(
            text=final_text or "\n".join(text_parts),
            tool_calls=tuple(tool_calls),
            stop_reason="end_turn",
        )

    def _query(self, prompt: str, options: Any) -> AsyncIterator[Any]:
        return self._sdk.query(prompt=prompt, options=options)

    def _to_sdk_tool(self, definition: ToolDefinition) -> Any:
        async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                result = definition.handler(arguments)
                if inspect.isawaitable(result):
                    result = await result
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, default=str),
                        }
                    ]
                }
            except Exception as exc:  # noqa: BLE001 - MCP tools report failures
                return {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }

        handler.__name__ = definition.name
        return self._sdk.tool(
            definition.name,
            definition.description,
            definition.input_schema,
        )(handler)

    @staticmethod
    def _render_prompt(messages: Sequence[Message]) -> str:
        lines: list[str] = []
        for message in messages:
            label = "User" if message.role == "user" else "Assistant"
            if message.text:
                lines.append(f"{label}: {message.text}")
            for result in message.tool_results:
                lines.append(f"Tool result ({result.tool_call_id}): {result.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _strip_mcp_prefix(name: str) -> str:
        return name.removeprefix("mcp__secure_agent__")
