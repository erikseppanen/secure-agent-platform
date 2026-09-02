from typing import Any

from mcp.server import MCPServer

from app.database import get_recent_incidents

# create a server
mcp = MCPServer(
    "Secure Agent Platform Tools",
    instructions=(
        "Tools for inspecting internal enterprise services and incidents. "
        "Call get_system_status for current simulated health and "
        "get_incidents for incident history stored in PostgreSQL."
    ),
)

# register a Python function
@mcp.tool()
def get_system_status(service: str) -> dict[str, Any]:
    """Get the current operational status and latency of an internal service."""

    services = {
        "billing": {
            "status": "healthy",
            "latency_ms": 87,
        },
        "authentication": {
            "status": "degraded",
            "latency_ms": 640,
        },
        "documents": {
            "status": "healthy",
            "latency_ms": 110,
        },
    }

    return services.get(
        service.lower(),
        {
            "status": "unknown",
            "message": f"No service named '{service}'",
        },
    )


@mcp.tool()
async def get_incidents(
    service: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get recent service incidents from PostgreSQL.

    Args:
        service: Optional service name such as authentication, billing, or documents.
        limit: Maximum number of incidents to return, from 1 through 50.
    """

    return await get_recent_incidents(service=service, limit=limit)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run("stdio")


if __name__ == "__main__":
    main()
