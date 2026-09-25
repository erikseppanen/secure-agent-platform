import json
import logging

import pytest

from app.config import get_settings
from app.observability import (
    JsonLineFormatter,
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


def test_json_line_formatter_keeps_structured_payload_nested() -> None:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="console message",
        args=(),
        exc_info=None,
    )
    record.structured_event = {
        "event": "test.event",
        "trace_id": "trace-123",
        "payload": {"nested": {"value": 7}},
    }

    output = json.loads(JsonLineFormatter().format(record))

    assert output["level"] == "INFO"
    assert output["logger"] == "app.test"
    assert output["event"] == "test.event"
    assert output["trace_id"] == "trace-123"
    assert output["payload"] == {"nested": {"value": 7}}
