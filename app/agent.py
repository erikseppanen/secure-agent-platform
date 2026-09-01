import json

from anthropic import Anthropic

from app.config import get_settings
from app.tools import TOOL_DEFINITIONS, execute_tool


settings = get_settings()

client = Anthropic(
    api_key=settings.anthropic_api_key,
    default_headers={
        "anthropic-workspace-id": settings.anthropic_workspace_id
    },
)

def run_agent(user_message: str) -> str:
    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    while True:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=(
                "You are an enterprise operations assistant. "
                "Use available tools when they are needed to "
                "answer factual questions about internal systems."
            ),
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text_blocks = [
                block.text
                for block in response.content
                if block.type == "text"
            ]

            return "\n".join(text_blocks)

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            try:
                result = execute_tool(
                    block.name,
                    block.input,
                )

                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }

            except Exception as exc:
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": str(exc),
                }

            tool_results.append(tool_result)

        messages.append(
            {
                "role": "user",
                "content": tool_results,
            }
        )