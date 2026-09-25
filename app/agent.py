import logging
from time import perf_counter
from typing import Any, Literal, TypedDict

from langgraph.types import Command

from app.agent_graph import AgentState
from app.checkpoint_runtime import get_agent_graph
from app.mcp_runtime import get_anthropic_tools
from app.observability import configure_logging, log_event, trace_context


configure_logging()
logger = logging.getLogger(__name__)


class NoPendingApprovalError(LookupError):
    """Raised when an approval decision targets a thread with no pending interrupt."""


class AgentRunResult(TypedDict):
    status: Literal["completed", "approval_required"]
    answer: str | None
    approval: dict[str, Any] | None


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def _result_from_graph(final_state: dict[str, Any]) -> AgentRunResult:
    interrupts = final_state.get("__interrupt__", ())
    if interrupts:
        current = interrupts[0]
        value = current.value
        approval = value if isinstance(value, dict) else {"message": str(value)}
        return {
            "status": "approval_required",
            "answer": None,
            "approval": approval,
        }

    return {
        "status": "completed",
        "answer": final_state.get("answer", ""),
        "approval": None,
    }


async def run_agent(user_message: str, thread_id: str) -> AgentRunResult:
    """Run one request through a persistent LangGraph thread."""

    with trace_context() as trace_id:
        started = perf_counter()
        log_event(
            logger,
            "agent.start",
            trace_id_created=trace_id,
            thread_id=thread_id,
            payload={"user_message": user_message},
        )

        tools = get_anthropic_tools()
        log_event(
            logger,
            "mcp.tools.available",
            thread_id=thread_id,
            tool_names=[tool["name"] for tool in tools],
        )

        input_state: AgentState = {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            "trace_id": trace_id,
            "stop_reason": None,
            "answer": "",
            "approval_granted": None,
        }
        config = _thread_config(thread_id)

        try:
            final_state = await get_agent_graph().ainvoke(
                input_state,
                config=config,
            )
            result = _result_from_graph(final_state)
            event = (
                "agent.interrupted"
                if result["status"] == "approval_required"
                else "agent.complete"
            )
            log_event(
                logger,
                event,
                thread_id=thread_id,
                duration_ms=round((perf_counter() - started) * 1000, 1),
                graph_turns=final_state.get("turn", 0),
                message_count=len(final_state.get("messages", [])),
                payload={
                    "answer": result["answer"],
                    "approval": result["approval"],
                },
            )
            return result
        except Exception:
            logger.exception(
                '{"event":"agent.error","trace_id":"%s","thread_id":"%s"}',
                trace_id,
                thread_id,
            )
            raise


async def resume_agent(
    thread_id: str,
    approved: bool,
) -> AgentRunResult:
    """Resume the single pending interrupt for a checkpointed graph thread."""

    with trace_context() as trace_id:
        started = perf_counter()
        log_event(
            logger,
            "agent.resume.start",
            trace_id_created=trace_id,
            thread_id=thread_id,
            approved=approved,
        )

        command = Command(resume=approved)
        config = _thread_config(thread_id)
        graph = get_agent_graph()

        snapshot = await graph.aget_state(config)
        if not snapshot.interrupts:
            log_event(
                logger,
                "agent.resume.no_pending_approval",
                thread_id=thread_id,
            )
            raise NoPendingApprovalError(thread_id)

        try:
            final_state = await graph.ainvoke(
                command,
                config=config,
            )
            result = _result_from_graph(final_state)
            log_event(
                logger,
                "agent.resume.complete",
                thread_id=thread_id,
                status=result["status"],
                duration_ms=round((perf_counter() - started) * 1000, 1),
                payload={
                    "answer": result["answer"],
                    "approval": result["approval"],
                },
            )
            return result
        except Exception:
            logger.exception(
                '{"event":"agent.resume.error","trace_id":"%s",'
                '"thread_id":"%s"}',
                trace_id,
                thread_id,
            )
            raise
