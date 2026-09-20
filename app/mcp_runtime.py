import logging
from contextlib import AsyncExitStack
from time import perf_counter
from typing import Any

from app.mcp_client import anthropic_tool_definition, create_mcp_client
from app.observability import configure_logging, log_event


configure_logging()
logger = logging.getLogger(__name__)

_exit_stack: AsyncExitStack | None = None
_mcp_client: Any | None = None
_anthropic_tools: list[dict[str, Any]] = []


async def start_mcp_runtime() -> None:
    """Start one MCP subprocess for the lifetime of the FastAPI application."""

    global _exit_stack, _mcp_client, _anthropic_tools

    if _mcp_client is not None:
        return

    started = perf_counter()
    log_event(logger, "mcp.runtime.start")

    stack = AsyncExitStack()
    try:
        client = await stack.enter_async_context(create_mcp_client())
        tools_response = await client.list_tools()
        tools = [
            anthropic_tool_definition(tool)
            for tool in tools_response.tools
        ]
    except Exception:
        await stack.aclose()
        logger.exception('{"event":"mcp.runtime.error"}')
        raise

    _exit_stack = stack
    _mcp_client = client
    _anthropic_tools = tools

    log_event(
        logger,
        "mcp.runtime.ready",
        duration_ms=round((perf_counter() - started) * 1000, 1),
        tool_names=[tool["name"] for tool in tools],
    )


async def stop_mcp_runtime() -> None:
    """Close the persistent MCP connection and its subprocess."""

    global _exit_stack, _mcp_client, _anthropic_tools

    stack = _exit_stack
    _exit_stack = None
    _mcp_client = None
    _anthropic_tools = []

    if stack is not None:
        await stack.aclose()
        log_event(logger, "mcp.runtime.stopped")


def get_mcp_client() -> Any:
    """Return the application-scoped MCP client."""

    if _mcp_client is None:
        raise RuntimeError("MCP runtime has not been started")
    return _mcp_client


def get_anthropic_tools() -> list[dict[str, Any]]:
    """Return tool definitions discovered once during application startup."""

    if _mcp_client is None:
        raise RuntimeError("MCP runtime has not been started")
    return _anthropic_tools
