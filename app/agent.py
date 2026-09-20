import logging
from time import perf_counter

from app.agent_graph import AgentState
from app.checkpoint_runtime import get_agent_graph
from app.mcp_runtime import get_anthropic_tools
from app.observability import configure_logging, log_event, trace_context


configure_logging()
logger = logging.getLogger(__name__)


async def run_agent(user_message: str, thread_id: str) -> str:
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
        }
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        try:
            final_state = await get_agent_graph().ainvoke(
                input_state,
                config=config,
            )
            answer = final_state["answer"]
            log_event(
                logger,
                "agent.complete",
                thread_id=thread_id,
                duration_ms=round((perf_counter() - started) * 1000, 1),
                graph_turns=final_state.get("turn", 0),
                message_count=len(final_state.get("messages", [])),
                payload={"answer": answer},
            )
            return answer
        except Exception:
            logger.exception(
                '{"event":"agent.error","trace_id":"%s","thread_id":"%s"}',
                trace_id,
                thread_id,
            )
            raise
