import json

from app.log_viewer import LOG_VIEWER_HTML, _json_event


def test_log_viewer_contains_live_event_stream() -> None:
    assert 'new EventSource("/logs/stream?tail=200")' in LOG_VIEWER_HTML
    assert "Expand all" in LOG_VIEWER_HTML
    assert "Collapse all" in LOG_VIEWER_HTML


def test_json_event_accepts_json_and_rejects_plain_text() -> None:
    event = {"event": "agent.start", "trace_id": "trace-123"}

    assert _json_event(json.dumps(event)) == json.dumps(event)
    assert _json_event("ordinary terminal output") is None
