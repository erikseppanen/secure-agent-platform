import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator
from uuid import uuid4

from app.config import get_settings


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def configure_logging() -> None:
    """Configure application logging once per process."""

    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def active_trace_id() -> str | None:
    """Return the current trace ID, including one inherited by an MCP subprocess."""

    return _trace_id.get() or os.getenv("SAP_TRACE_ID")


def current_trace_id() -> str:
    """Return a printable trace ID for log records."""

    return active_trace_id() or "untraced"


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Create a trace context that follows one agent request."""

    resolved = trace_id or uuid4().hex[:12]
    token = _trace_id.set(resolved)
    try:
        yield resolved
    finally:
        _trace_id.reset(token)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def payload_preview(value: Any) -> str:
    """Serialize and cap potentially large trace payloads."""

    settings = get_settings()
    text = json.dumps(_jsonable(value), default=str, ensure_ascii=False)
    if len(text) <= settings.trace_max_chars:
        return text
    return text[: settings.trace_max_chars] + "...<truncated>"


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    payload: Any | None = None,
    **fields: Any,
) -> None:
    """Write one structured trace event without logging credentials."""

    record: dict[str, Any] = {
        "event": event,
        "trace_id": current_trace_id(),
        **fields,
    }

    if payload is not None and get_settings().trace_payloads:
        record["payload"] = payload_preview(payload)

    logger.log(
        level,
        json.dumps(_jsonable(record), default=str, ensure_ascii=False),
    )
