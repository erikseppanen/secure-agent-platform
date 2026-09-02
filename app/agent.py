from anthropic import AsyncAnthropic

from app.config import get_settings
from app.mcp_client import (
    anthropic_tool_definition,
    create_mcp_client,
    tool_result_text,
)


settings = get_settings()

client = AsyncAnthropic(
    api_key=settings.anthropic_api_key,
    default_headers={
        "anthropic-workspace-id": settings.anthropic_workspace_id
    },
)


async def run_agent(user_message: str) -> str:
    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    # create client
    async with create_mcp_client() as mcp_client:
        # list_tools() is one of MCP SDK's Client methods
        # MCP handles returning the tools from the available methods
        tools_response = await mcp_client.list_tools()
        tools = [
            anthropic_tool_definition(tool)
            for tool in tools_response.tools
        ]

        while True:
            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1024,
                system=(
                    "You are an enterprise operations assistant. "
                    "Use available tools when they are needed to "
                    "answer factual questions about internal systems."
                ),
                tools=tools,
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

                # call tool (through mcp client)
                result = await mcp_client.call_tool(
                    block.name,
                    block.input,
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "is_error": result.is_error,
                        "content": tool_result_text(result),
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )
