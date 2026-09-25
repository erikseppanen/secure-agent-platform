import pytest
from mcp import Client

from app.mcp_server import get_system_status, mcp, restart_service


@pytest.mark.asyncio
async def test_mcp_server_lists_expected_tools() -> None:
    async with Client(mcp) as client:
        response = await client.list_tools()

    names = {tool.name for tool in response.tools}
    assert "get_system_status" in names
    assert "get_incidents" in names
    assert "search_documents" in names
    assert "restart_service" in names


@pytest.mark.asyncio
async def test_mcp_server_calls_system_status_tool() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_system_status",
            {"service": "authentication"},
        )

    assert result.is_error is False
    assert result.content


def test_system_status_data() -> None:
    assert get_system_status("authentication") == {
        "status": "degraded",
        "latency_ms": 640,
    }


def test_restart_service_is_simulated() -> None:
    assert restart_service("authentication") == {
        "service": "authentication",
        "action": "restart",
        "status": "completed",
        "simulated": True,
    }
