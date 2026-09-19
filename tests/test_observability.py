import os

import pytest

from app.config import get_settings
from app.mcp_client import _server_parameters
from app.observability import (
    active_trace_id,
    payload_preview,
    trace_context,
)


@pytest.fixture(autouse=True)
def configure_test_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "test-workspace")
    monkeypatch.setenv("TRACE_MAX_CHARS", "20")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_trace_context_sets_and_restores_trace_id() -> None:
    before = active_trace_id()

    with trace_context("trace-123"):
        assert active_trace_id() == "trace-123"

    assert active_trace_id() == before


def test_payload_preview_truncates_large_payloads() -> None:
    preview = payload_preview({"message": "x" * 100})

    assert preview.endswith("...<truncated>")
    assert len(preview) < 100


def test_mcp_subprocess_receives_current_trace_id() -> None:
    with trace_context("trace-456"):
        parameters = _server_parameters()

    assert parameters.env is not None
    assert parameters.env["SAP_TRACE_ID"] == "trace-456"
    assert parameters.env.get("PATH") == os.environ.get("PATH")
