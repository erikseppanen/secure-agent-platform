from types import SimpleNamespace

import pytest

import app.database as database


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.calls = []

    async def fetch(self, query: str, *args):
        self.calls.append((query, args))
        return [
            {
                "id": 1,
                "service": "authentication",
                "severity": "high",
                "status": "monitoring",
                "summary": "Elevated login latency.",
                "started_at": "2026-09-02T12:00:00Z",
            }
        ]

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_get_recent_incidents_filters_and_clamps_limit(monkeypatch) -> None:
    connection = FakeConnection()

    async def fake_connect(url: str):
        assert url == "postgresql://example"
        return connection

    monkeypatch.setattr(
        database,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://example"),
    )
    monkeypatch.setattr(database.asyncpg, "connect", fake_connect)

    rows = await database.get_recent_incidents(
        service="Authentication",
        limit=500,
    )

    assert rows[0]["service"] == "authentication"
    assert connection.calls[0][1] == ("authentication", 50)
    assert connection.closed is True
