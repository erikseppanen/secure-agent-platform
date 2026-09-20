import logging
from time import perf_counter

from app.agent_graph import AgentState, agent_graph
from app.mcp_runtime import get_anthropic_tools
from app.observability import configure_logging, log_event, trace_context


configure_logging()
logger = logging.getLogger(__name__)


async def run_agent(user_message: str) -> str:
    """Run one request through the compiled LangGraph agent."""

    with trace_context() as trace_id:
        started = perf_counter()
        log_event(
            logger,
            "agent.start",
            trace_id_created=trace_id,
            payload={"user_message": user_message},
        )

        tools = get_anthropic_tools()
        log_event(
            logger,
            "mcp.tools.available",
            tool_names=[tool["name"] for tool in tools],
        )

        initial_state: AgentState = {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            "trace_id": trace_id,
            "turn": 0,
            "response": None,
            "answer": "",
        }

        try:
            final_state = await agent_graph.ainvoke(initial_state)
            answer = final_state["answer"]
            log_event(
                logger,
                "agent.complete",
                duration_ms=round((perf_counter() - started) * 1000, 1),
                graph_turns=final_state["turn"],
                payload={"answer": answer},
            )
            return answer
        except Exception:
            logger.exception(
                '{"event":"agent.error","trace_id":"%s"}',
                trace_id,
            )
            raise
