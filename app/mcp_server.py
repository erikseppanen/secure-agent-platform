from typing import Any

from mcp.server import MCPServer

# create a server
mcp = MCPServer(
    "Secure Agent Platform Tools",
    instructions=(
        "Tools for inspecting internal enterprise services. "
        "Call get_system_status when current service health is needed."
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

# using the decorator @mcp.tool(),
# MCP can use: get_system_status(service: str)
# to derive:
# {
#    "name": "get_system_status",
#    "description": "Get the current operational status...",
#    "inputSchema": {
#        "type": "object",
#        "properties": {
#            "service": {
#                "type": "string"
#            }
#        },
#        "required": ["service"]
#    }
# }

    return services.get(
        service.lower(),
        {
            "status": "unknown",
            "message": f"No service named '{service}'",
        },
    )


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run("stdio")


if __name__ == "__main__":
    main()
