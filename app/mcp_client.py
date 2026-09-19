import json
import os
import sys
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, Tool

from app.observability import active_trace_id


def _server_parameters() -> StdioServerParameters:
    """Describe the MCP subprocess and propagate the current trace ID."""

    env = os.environ.copy()
    trace_id = active_trace_id()
    if trace_id is not None:
        env["SAP_TRACE_ID"] = trace_id

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=env,
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
    return Client(stdio_client(_server_parameters()))
