from collections.abc import Sequence
from typing import Any

from app.llm.base import LLMProvider
from app.llm.models import Message, ModelResponse, ToolCall, ToolDefinition


class AnthropicApiProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        workspace_id: str | None = None,
        max_tokens: int = 1024,
        system_prompt: str,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic_api")

        if client is None:
            from anthropic import AsyncAnthropic

            headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
            client = AsyncAnthropic(api_key=api_key, default_headers=headers)

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    async def run(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            tools=[tool.anthropic_schema() for tool in tools],
            messages=self._serialize_messages(messages),
        )

        text: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return ModelResponse(
            text="\n".join(text),
            tool_calls=tuple(tool_calls),
            stop_reason=response.stop_reason or "end_turn",
        )

    @staticmethod
    def _serialize_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []

        for message in messages:
            content: str | list[dict[str, Any]]

            if message.tool_results:
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        **({"is_error": True} if result.is_error else {}),
                    }
                    for result in message.tool_results
                ]
            elif message.tool_calls:
                blocks: list[dict[str, Any]] = []
                if message.text:
                    blocks.append({"type": "text", "text": message.text})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                content = blocks
            else:
                content = message.text

            serialized.append({"role": message.role, "content": content})

        return serialized
