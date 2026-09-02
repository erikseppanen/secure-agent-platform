from collections.abc import Sequence
from typing import Any

import asyncpg

from app.config import get_settings


async def get_recent_incidents(
    service: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return recent incidents, optionally filtered by service."""

    safe_limit = max(1, min(limit, 50))
    settings = get_settings()

    connection = await asyncpg.connect(settings.database_url)
    try:
        rows: Sequence[asyncpg.Record]

        if service:
            rows = await connection.fetch(
                """
                SELECT id, service, severity, status, summary, started_at
                FROM incidents
                WHERE service = $1
                ORDER BY started_at DESC
                LIMIT $2
                """,
                service.lower(),
                safe_limit,
            )
        else:
            rows = await connection.fetch(
                """
                SELECT id, service, severity, status, summary, started_at
                FROM incidents
                ORDER BY started_at DESC
                LIMIT $1
                """,
                safe_limit,
            )

        return [dict(row) for row in rows]
    finally:
        await connection.close()
