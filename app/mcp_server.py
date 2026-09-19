import logging
from time import perf_counter
from typing import Any

from mcp.server import MCPServer

from app.database import get_recent_incidents
from app.observability import configure_logging, log_event
from app.rag import hybrid_search_documents


configure_logging()
logger = logging.getLogger(__name__)

mcp = MCPServer(
    "Secure Agent Platform Tools",
    instructions=(
        "Tools for inspecting internal enterprise services, incidents, and "
        "documentation. Call get_system_status for current simulated health, "
        "get_incidents for incident history, and search_documents for runbooks, "
        "procedures, policies, and other internal knowledge."
    ),
)


@mcp.tool()
def get_system_status(service: str) -> dict[str, Any]:
    """Get the current operational status and latency of an internal service."""

    log_event(
        logger,
        "mcp.server.get_system_status.request",
        payload={"service": service},
    )

    services = {
        "billing": {"status": "healthy", "latency_ms": 87},
        "authentication": {"status": "degraded", "latency_ms": 640},
        "documents": {"status": "healthy", "latency_ms": 110},
    }

    result = services.get(
        service.lower(),
        {"status": "unknown", "message": f"No service named '{service}'"},
    )
    log_event(
        logger,
        "mcp.server.get_system_status.response",
        payload={"result": result},
    )
    return result


@mcp.tool()
async def get_incidents(
    service: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get recent service incidents from PostgreSQL."""

    started = perf_counter()
    log_event(
        logger,
        "mcp.server.get_incidents.request",
        payload={"service": service, "limit": limit},
    )
    result = await get_recent_incidents(service=service, limit=limit)
    log_event(
        logger,
        "mcp.server.get_incidents.response",
        duration_ms=round((perf_counter() - started) * 1000, 1),
        result_count=len(result),
        payload={"result": result},
    )
    return result


@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    service: str | None = None,
    document_type: str | None = None,
    environment: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid-search internal documentation with optional metadata filters."""

    started = perf_counter()
    arguments = {
        "query": query,
        "limit": limit,
        "service": service,
        "document_type": document_type,
        "environment": environment,
    }
    log_event(
        logger,
        "rag.request",
        payload=arguments,
    )

    result = await hybrid_search_documents(**arguments)

    log_event(
        logger,
        "rag.response",
        duration_ms=round((perf_counter() - started) * 1000, 1),
        result_count=len(result),
        payload={"results": result},
    )
    return result


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run("stdio")


if __name__ == "__main__":
    main()
