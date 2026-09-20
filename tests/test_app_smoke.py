"""Smoke test for the real FastAPI app: import, startup, and shutdown.

No other test imports `app.main` or `app.checkpoint_runtime` — the rest of the
suite exercises `app.agent_graph` directly with an in-memory checkpointer. That
left the real Postgres-backed startup path (and anything else only reachable
through `app.main`) unverified: a missing `psycopg` pq backend once broke
`uv run uvicorn app.main:app` while `pytest` stayed green. This test imports
the app for real and runs its lifespan, so an import- or startup-time break
fails here too.

Requires the local Postgres (`docker compose up`) and the local embedding/
reranker models to already be cached, same as tests/test_rag.py and
tests/test_reranker.py.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(autouse=True)
def configure_test_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "test-workspace")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_app_starts_up_and_serves_health() -> None:
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
