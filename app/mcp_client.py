import json
import sys
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, Tool

# Start another Python process by running python -m app.mcp_server
# (mcp_server.py will be running in a separate process)
SERVER_PARAMETERS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "app.mcp_server"],
)


def anthropic_tool_definition(tool: Tool) -> dict[str, Any]:
    """Translate an MCP tool definition into Anthropic's tool schema."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }


def tool_result_text(result: CallToolResult) -> str:
    """Convert an MCP tool result into text suitable for an LLM tool_result."""
    text_parts = [
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    ]
    if text_parts:
        return "\n".join(text_parts)

    if result.structured_content is not None:
        return json.dumps(result.structured_content)

    return ""


def create_mcp_client() -> Client:
    """Create a client for the local stdio MCP server."""
    # the MCP client starts the server process and communicates with it over its standard input and output.
    return Client(stdio_client(SERVER_PARAMETERS))
