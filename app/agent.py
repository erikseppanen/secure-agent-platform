import inspect
import json

from app.config import get_settings
from app.llm.base import LLMProvider
from app.llm.factory import create_provider
from app.llm.models import Message, ToolResult
from app.tools import TOOLS


async def run_agent(
    user_message: str,
    provider: LLMProvider | None = None,
) -> str:
    settings = get_settings()
    provider = provider or create_provider(settings)
    messages = [Message(role="user", text=user_message)]

    for _ in range(settings.agent_max_steps):
        response = await provider.run(messages, TOOLS)

        if response.stop_reason != "tool_use" or not response.tool_calls:
            return response.text

        messages.append(
            Message(
                role="assistant",
                text=response.text,
                tool_calls=response.tool_calls,
            )
        )

        results: list[ToolResult] = []
        tools_by_name = {tool.name: tool for tool in TOOLS}

        for call in response.tool_calls:
            try:
                definition = tools_by_name[call.name]
                result = definition.handler(call.arguments)
                if inspect.isawaitable(result):
                    result = await result
                tool_result = ToolResult(
                    tool_call_id=call.id,
                    content=json.dumps(result, default=str),
                )
            except Exception as exc:  # noqa: BLE001 - tool failures become results
                tool_result = ToolResult(
                    tool_call_id=call.id,
                    content=str(exc),
                    is_error=True,
                )
            results.append(tool_result)

        messages.append(Message(role="user", tool_results=tuple(results)))

    raise RuntimeError(
        f"Agent exceeded the configured limit of {settings.agent_max_steps} steps"
    )
