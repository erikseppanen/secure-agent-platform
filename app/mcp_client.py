import json
import os
import sys
from copy import deepcopy
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, Tool


INTERNAL_TRACE_ARGUMENT = "trace_id_internal"


def _server_parameters() -> StdioServerParameters:
    """Describe the long-lived local MCP subprocess."""

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=os.environ.copy(),
    )


def _public_input_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    """Hide transport-only arguments from the schema exposed to Claude."""

    schema = deepcopy(input_schema)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop(INTERNAL_TRACE_ARGUMENT, None)

    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [
            name
            for name in required
            if name != INTERNAL_TRACE_ARGUMENT
        ]

    return schema


def anthropic_tool_definition(tool: Tool) -> dict[str, Any]:
    """Translate an MCP tool definition into Anthropic's public tool schema."""

    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": _public_input_schema(tool.input_schema),
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
    """Create the client that owns the local stdio MCP subprocess."""

    return Client(stdio_client(_server_parameters()))
