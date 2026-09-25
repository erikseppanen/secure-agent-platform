import logging
import operator
from functools import lru_cache
from time import perf_counter
from typing import Annotated, Any, TypedDict

from anthropic import AsyncAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

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

APPROVAL_REQUIRED_TOOLS = frozenset({"restart_service"})


class AgentState(TypedDict, total=False):
    """State persisted and passed between LangGraph nodes."""

    messages: Annotated[list[dict[str, Any]], operator.add]
    trace_id: str
    turn: int
    stop_reason: str | None
    answer: str
    approval_granted: bool | None


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


def _content_dicts(content: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic response blocks into checkpoint-safe dictionaries."""

    return [
        block.model_dump() if hasattr(block, "model_dump") else dict(block)
        for block in content
    ]


def _latest_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Return tool_use blocks from the latest assistant message."""

    messages = state.get("messages", [])
    if not messages:
        return []

    assistant_message = messages[-1]
    if assistant_message.get("role") != "assistant":
        return []

    content = assistant_message.get("content")
    if not isinstance(content, list):
        return []

    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


async def call_model(state: AgentState) -> dict[str, Any]:
    """Ask Claude what to do next and append its response to graph state."""

    settings = get_settings()
    tools = get_anthropic_tools()
    turn = state.get("turn", 0) + 1
    started = perf_counter()

    log_event(
        logger,
        "claude.request",
        model=settings.anthropic_model,
        turn=turn,
        message_count=len(state.get("messages", [])),
        payload={
            "messages": state.get("messages", []),
            "tools": tools,
        },
    )

    response = await get_anthropic_client().messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=state.get("messages", []),
    )

    usage = getattr(response, "usage", None)
    stop_reason = response.stop_reason
    content = _content_dicts(response.content)

    log_event(
        logger,
        "claude.response",
        turn=turn,
        stop_reason=stop_reason,
        duration_ms=round((perf_counter() - started) * 1000, 1),
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        payload={"content": content},
    )

    text_blocks = [
        block["text"]
        for block in content
        if block.get("type") == "text"
    ]
    answer = "\n".join(text_blocks) if stop_reason != "tool_use" else ""

    return {
        "messages": [
            {
                "role": "assistant",
                "content": content,
            }
        ],
        "turn": turn,
        "stop_reason": stop_reason,
        "answer": answer,
    }


def route_after_model(state: AgentState) -> str:
    """Route Claude tool requests through policy before execution."""

    stop_reason = state.get("stop_reason")
    if stop_reason != "tool_use":
        destination = END
    else:
        tool_calls = _latest_tool_calls(state)
        requires_approval = any(
            block.get("name") in APPROVAL_REQUIRED_TOOLS
            for block in tool_calls
        )
        destination = "approval_gate" if requires_approval else "execute_tools"

    log_event(
        logger,
        "langgraph.route",
        from_node="call_model",
        to_node=destination,
        stop_reason=stop_reason,
    )
    return destination


def approval_gate(state: AgentState) -> dict[str, Any]:
    """Interrupt before sensitive tools and wait for a human decision."""

    sensitive_tools = [
        {
            "tool_use_id": block["id"],
            "name": block["name"],
            "arguments": block.get("input", {}),
        }
        for block in _latest_tool_calls(state)
        if block.get("name") in APPROVAL_REQUIRED_TOOLS
    ]
    if not sensitive_tools:
        return {"approval_granted": True}

    approval_request = {
        "type": "tool_approval",
        "message": "Human approval is required before executing sensitive tools.",
        "tools": sensitive_tools,
    }
    log_event(
        logger,
        "agent.approval.required",
        tool_names=[tool["name"] for tool in sensitive_tools],
        payload=approval_request,
    )

    decision = interrupt(approval_request)
    if not isinstance(decision, bool):
        raise RuntimeError("Tool approval decision must be a boolean")

    log_event(
        logger,
        "agent.approval.decision",
        approved=decision,
        tool_names=[tool["name"] for tool in sensitive_tools],
    )
    return {"approval_granted": decision}


async def execute_tools(state: AgentState) -> dict[str, Any]:
    """Execute Claude tool requests through the persistent MCP client."""

    tool_calls = _latest_tool_calls(state)
    if not tool_calls:
        raise RuntimeError("execute_tools requires an assistant tool request")

    mcp_client = get_mcp_client()
    tool_results = []
    approval_granted = state.get("approval_granted")

    for block in tool_calls:
        tool_name = block["name"]
        tool_use_id = block["id"]
        tool_input = block.get("input", {})

        if (
            tool_name in APPROVAL_REQUIRED_TOOLS
            and approval_granted is not True
        ):
            rejection = "Human approval denied; the tool was not executed."
            log_event(
                logger,
                "mcp.tool.rejected",
                tool=tool_name,
                tool_use_id=tool_use_id,
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": rejection,
                }
            )
            continue

        tool_started = perf_counter()
        log_event(
            logger,
            "mcp.tool.request",
            tool=tool_name,
            tool_use_id=tool_use_id,
            payload={"arguments": tool_input},
        )

        tool_arguments = dict(tool_input)
        tool_arguments[INTERNAL_TRACE_ARGUMENT] = state["trace_id"]
        result = await mcp_client.call_tool(
            tool_name,
            tool_arguments,
        )
        result_text = tool_result_text(result)

        log_event(
            logger,
            "mcp.tool.response",
            tool=tool_name,
            tool_use_id=tool_use_id,
            is_error=result.is_error,
            duration_ms=round((perf_counter() - tool_started) * 1000, 1),
            payload={"result": result_text},
        )

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "is_error": result.is_error,
                "content": result_text,
            }
        )

    return {
        "messages": [
            {
                "role": "user",
                "content": tool_results,
            }
        ],
        "stop_reason": None,
        "approval_granted": None,
    }


def build_agent_graph(checkpointer: Any | None = None):
    """Compile the agent graph, optionally with durable checkpoint storage."""

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("approval_gate", "execute_tools")
    builder.add_edge("execute_tools", "call_model")
    return builder.compile(checkpointer=checkpointer)
