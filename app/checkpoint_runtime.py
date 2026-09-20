import logging
import os
from contextlib import AsyncExitStack
from time import perf_counter
from typing import Any

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent_graph import build_agent_graph
from app.config import get_settings
from app.observability import configure_logging, log_event


configure_logging()
logger = logging.getLogger(__name__)

_exit_stack: AsyncExitStack | None = None
_agent_graph: Any | None = None


async def start_checkpoint_runtime() -> None:
    """Start the PostgreSQL checkpointer and compile the persistent graph."""

    global _exit_stack, _agent_graph

    if _agent_graph is not None:
        return

    started = perf_counter()
    log_event(logger, "checkpoint.runtime.start")

    stack = AsyncExitStack()
    try:
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(
                get_settings().database_url,
            )
        )
        await checkpointer.setup()
        graph = build_agent_graph(checkpointer=checkpointer)
    except Exception:
        await stack.aclose()
        logger.exception('{"event":"checkpoint.runtime.error"}')
        raise

    _exit_stack = stack
    _agent_graph = graph

    log_event(
        logger,
        "checkpoint.runtime.ready",
        duration_ms=round((perf_counter() - started) * 1000, 1),
    )


async def stop_checkpoint_runtime() -> None:
    """Close the PostgreSQL checkpointer connection."""

    global _exit_stack, _agent_graph

    stack = _exit_stack
    _exit_stack = None
    _agent_graph = None

    if stack is not None:
        await stack.aclose()
        log_event(logger, "checkpoint.runtime.stopped")


def get_agent_graph() -> Any:
    """Return the application-scoped graph compiled with its checkpointer."""

    if _agent_graph is None:
        raise RuntimeError("Checkpoint runtime has not been started")
    return _agent_graph
