import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from app.config import get_settings


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_FILE_HANDLER_MARKER = "_sap_json_file_handler"


class JsonLineFormatter(logging.Formatter):
    """Serialize one log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "structured_event", None)

        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "process": record.process,
        }

        if isinstance(event, dict):
            data.update(_jsonable(event))
        else:
            data["message"] = record.getMessage()

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            _jsonable(data),
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def configure_logging() -> None:
    """Configure console logging and a structured JSONL file once per process."""

    settings = get_settings()
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if any(
        getattr(handler, _FILE_HANDLER_MARKER, False)
        for handler in root_logger.handlers
    ):
        return

    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonLineFormatter())
    setattr(file_handler, _FILE_HANDLER_MARKER, True)
    root_logger.addHandler(file_handler)


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


def _structured_payload(value: Any) -> Any:
    """Keep payloads nested in JSONL when the configured preview is valid JSON."""

    preview = payload_preview(value)
    try:
        return json.loads(preview)
    except json.JSONDecodeError:
        return preview


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

    console_record = dict(record)
    if payload is not None and get_settings().trace_payloads:
        console_record["payload"] = payload_preview(payload)
        record["payload"] = _structured_payload(payload)

    logger.log(
        level,
        json.dumps(
            _jsonable(console_record),
            default=str,
            ensure_ascii=False,
        ),
        extra={"structured_event": _jsonable(record)},
    )
