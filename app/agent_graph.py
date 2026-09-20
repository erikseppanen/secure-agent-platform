import logging
from functools import lru_cache
from time import perf_counter
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.mcp_client import INTERNAL_TRACE_ARGUMENT, tool_result_text
from app.mcp_runtime import get_anthropic_tools, get_mcp_client
from app.observability import configure_logging, log_event


configure_logging()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise operations assistant. "
    "Use available tools when they are needed to "
    "answer factual questions about internal systems."
)


class AgentState(TypedDict):
    """State passed between LangGraph nodes during one agent run."""

    messages: list[dict[str, Any]]
    trace_id: str
    turn: int
    response: Any | None
    answer: str


@lru_cache
def get_anthropic_client() -> AsyncAnthropic:
    """Create the Anthropic client lazily so graph construction stays testable."""

    settings = get_settings()
    return AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        default_headers={
            "anthropic-workspace-id": settings.anthropic_workspace_id
        },
    )


async def call_model(state: AgentState) -> dict[str, Any]:
    """Ask Claude what to do next and append its response to graph state."""

    settings = get_settings()
    tools = get_anthropic_tools()
    turn = state["turn"] + 1
    started = perf_counter()

    log_event(
        logger,
        "claude.request",
        model=settings.anthropic_model,
        turn=turn,
        message_count=len(state["messages"]),
        payload={
            "messages": state["messages"],
            "tools": tools,
        },
    )

    response = await get_anthropic_client().messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=state["messages"],
    )

    usage = getattr(response, "usage", None)
    log_event(
        logger,
        "claude.response",
        turn=turn,
        stop_reason=response.stop_reason,
        duration_ms=round((perf_counter() - started) * 1000, 1),
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        payload={"content": response.content},
    )

    text_blocks = [
        block.text
        for block in response.content
        if block.type == "text"
    ]
    answer = "\n".join(text_blocks) if response.stop_reason != "tool_use" else ""

    return {
        "messages": [
            *state["messages"],
            {
                "role": "assistant",
                "content": response.content,
            },
        ],
        "turn": turn,
        "response": response,
        "answer": answer,
    }


def route_after_model(state: AgentState) -> str:
    """Route to tools when Claude requested one; otherwise finish the graph."""

    response = state["response"]
    destination = (
        "execute_tools"
        if response is not None and response.stop_reason == "tool_use"
        else END
    )

    log_event(
        logger,
        "langgraph.route",
        from_node="call_model",
        to_node=destination,
        stop_reason=getattr(response, "stop_reason", None),
    )
    return destination


async def execute_tools(state: AgentState) -> dict[str, Any]:
    """Execute Claude tool requests through the persistent MCP client."""

    response = state["response"]
    if response is None:
        raise RuntimeError("execute_tools requires a Claude response")

    mcp_client = get_mcp_client()
    tool_results = []

    for block in response.content:
        if block.type != "tool_use":
            continue

        tool_started = perf_counter()
        log_event(
            logger,
            "mcp.tool.request",
            tool=block.name,
            tool_use_id=block.id,
            payload={"arguments": block.input},
        )

        tool_arguments = dict(block.input)
        tool_arguments[INTERNAL_TRACE_ARGUMENT] = state["trace_id"]
        result = await mcp_client.call_tool(
            block.name,
            tool_arguments,
        )
        result_text = tool_result_text(result)

        log_event(
            logger,
            "mcp.tool.response",
            tool=block.name,
            tool_use_id=block.id,
            is_error=result.is_error,
            duration_ms=round((perf_counter() - tool_started) * 1000, 1),
            payload={"result": result_text},
        )

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": result.is_error,
                "content": result_text,
            }
        )

    return {
        "messages": [
            *state["messages"],
            {
                "role": "user",
                "content": tool_results,
            },
        ],
        "response": None,
    }


def build_agent_graph():
    """Compile the agent's explicit Claude -> tools -> Claude state machine."""

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("execute_tools", "call_model")
    return builder.compile()


agent_graph = build_agent_graph()
