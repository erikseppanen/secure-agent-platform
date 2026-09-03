from typing import Any

from mcp.server import MCPServer

from app.database import get_recent_incidents
from app.rag import hybrid_search_documents

# create a server
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

    services = {
        "billing": {"status": "healthy", "latency_ms": 87},
        "authentication": {"status": "degraded", "latency_ms": 640},
        "documents": {"status": "healthy", "latency_ms": 110},
    }

    return services.get(
        service.lower(),
        {"status": "unknown", "message": f"No service named '{service}'"},
    )


@mcp.tool()
async def get_incidents(
    service: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get recent service incidents from PostgreSQL."""

    return await get_recent_incidents(service=service, limit=limit)


@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    service: str | None = None,
    document_type: str | None = None,
    environment: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid-search internal documentation with optional metadata filters.

    Args:
        query: Natural-language description or exact terms to retrieve.
        limit: Maximum matching chunks to return, from 1 through 10.
        service: Optional service scope, such as authentication or billing.
        document_type: Optional type such as runbook or policy.
        environment: Optional environment such as production or staging.
    """

    return await hybrid_search_documents(
        query=query,
        limit=limit,
        service=service,
        document_type=document_type,
        environment=environment,
    )


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run("stdio")


if __name__ == "__main__":
    main()
