from typing import Any

from app.llm.models import ToolDefinition


def get_system_status(service: str) -> dict[str, Any]:
    """Return simulated status information for an internal service."""

    services = {
        "billing": {"status": "healthy", "latency_ms": 87},
        "authentication": {"status": "degraded", "latency_ms": 640},
        "documents": {"status": "healthy", "latency_ms": 110},
    }

    return services.get(
        service.lower(),
        {
            "status": "unknown",
            "message": f"No service named '{service}'",
        },
    )


def _get_system_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_system_status(**arguments)


TOOLS = [
    ToolDefinition(
        name="get_system_status",
        description=(
            "Get the current operational status and latency of an internal "
            "enterprise service."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Service name, such as billing, authentication, or documents."
                    ),
                }
            },
            "required": ["service"],
        },
        handler=_get_system_status,
    )
]


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    for definition in TOOLS:
        if definition.name == name:
            return definition.handler(arguments)

    raise ValueError(f"Unknown tool: {name}")
