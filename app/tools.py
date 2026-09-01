from typing import Any


def get_system_status(service: str) -> dict[str, Any]:
    """Return simulated status information for an internal service."""

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


TOOL_DEFINITIONS = [
    {
        "name": "get_system_status",
        "description": (
            "Get the current operational status and latency "
            "of an internal enterprise service."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Service name, such as billing, "
                        "authentication, or documents."
                    ),
                }
            },
            "required": ["service"],
        },
    }
]


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "get_system_status":
        return get_system_status(**arguments)

    raise ValueError(f"Unknown tool: {name}")