import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import get_settings


router = APIRouter()

LOG_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Secure Agent Platform Logs</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        "Liberation Mono", "Courier New", monospace;
    }
    body {
      margin: 0;
      background: #111827;
      color: #e5e7eb;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
      padding: 0.75rem 1rem;
      background: #0f172a;
      border-bottom: 1px solid #334155;
    }
    header h1 {
      margin: 0 1rem 0 0;
      font: 600 1rem system-ui, sans-serif;
    }
    input, select, button {
      font: inherit;
      padding: 0.35rem 0.5rem;
      border-radius: 0.35rem;
      border: 1px solid #475569;
      background: #111827;
      color: #e5e7eb;
    }
    button { cursor: pointer; }
    label {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font: 0.8rem system-ui, sans-serif;
    }
    #status {
      margin-left: auto;
      font: 0.8rem system-ui, sans-serif;
      color: #94a3b8;
    }
    #status.connected { color: #86efac; }
    #status.error { color: #fca5a5; }
    #logs { padding: 0.75rem; }
    details.entry {
      margin-bottom: 0.45rem;
      border: 1px solid #334155;
      border-radius: 0.4rem;
      background: #0f172a;
      overflow: hidden;
    }
    details.entry > summary {
      display: flex;
      gap: 0.65rem;
      align-items: baseline;
      cursor: pointer;
      list-style: none;
      padding: 0.45rem 0.6rem;
      white-space: nowrap;
      overflow: hidden;
    }
    details.entry > summary::-webkit-details-marker { display: none; }
    details.entry > summary::before {
      content: "▸";
      color: #94a3b8;
      flex: none;
    }
    details.entry[open] > summary::before { content: "▾"; }
    .timestamp { color: #94a3b8; }
    .level { font-weight: 700; min-width: 4.5rem; }
    .level.ERROR, .level.CRITICAL { color: #fca5a5; }
    .level.WARNING { color: #fde68a; }
    .level.INFO { color: #93c5fd; }
    .event { color: #c4b5fd; font-weight: 700; }
    .logger { color: #94a3b8; overflow: hidden; text-overflow: ellipsis; }
    .meta { color: #67e8f9; overflow: hidden; text-overflow: ellipsis; }
    .tree {
      border-top: 1px solid #334155;
      padding: 0.55rem 0.75rem 0.7rem 1.5rem;
      font-size: 0.84rem;
      line-height: 1.45;
    }
    .tree details { margin-left: 1rem; }
    .tree summary { cursor: pointer; color: #d8b4fe; }
    .leaf {
      margin-left: 1rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .key { color: #7dd3fc; }
    .string { color: #86efac; }
    .number { color: #fcd34d; }
    .boolean { color: #f9a8d4; }
    .null { color: #94a3b8; }
    .hidden { display: none !important; }
    .empty {
      padding: 2rem;
      text-align: center;
      color: #94a3b8;
      font: 0.9rem system-ui, sans-serif;
    }
  </style>
</head>
<body>
  <header>
    <h1>Secure Agent Platform Logs</h1>
    <select id="level">
      <option value="">All levels</option>
      <option>DEBUG</option>
      <option>INFO</option>
      <option>WARNING</option>
      <option>ERROR</option>
      <option>CRITICAL</option>
    </select>
    <input id="search" type="search" placeholder="filter text / trace / thread">
    <label><input id="autoscroll" type="checkbox" checked> auto-scroll</label>
    <button id="expand">Expand all</button>
    <button id="collapse">Collapse all</button>
    <button id="clear">Clear view</button>
    <span id="status">connecting…</span>
  </header>
  <main id="logs">
    <div class="empty">Waiting for log events…</div>
  </main>

  <script>
    const logs = document.getElementById("logs");
    const levelFilter = document.getElementById("level");
    const searchFilter = document.getElementById("search");
    const autoscroll = document.getElementById("autoscroll");
    const status = document.getElementById("status");
    const maxEntries = 1000;

    function scalarNode(key, value) {
      const row = document.createElement("div");
      row.className = "leaf";

      const keySpan = document.createElement("span");
      keySpan.className = "key";
      keySpan.textContent = key + ": ";
      row.appendChild(keySpan);

      const valueSpan = document.createElement("span");
      if (value === null) {
        valueSpan.className = "null";
        valueSpan.textContent = "null";
      } else if (typeof value === "string") {
        valueSpan.className = "string";
        valueSpan.textContent = JSON.stringify(value);
      } else if (typeof value === "number") {
        valueSpan.className = "number";
        valueSpan.textContent = String(value);
      } else if (typeof value === "boolean") {
        valueSpan.className = "boolean";
        valueSpan.textContent = String(value);
      } else {
        valueSpan.textContent = String(value);
      }
      row.appendChild(valueSpan);
      return row;
    }

    function treeNode(key, value, depth = 0) {
      if (value === null || typeof value !== "object") {
        return scalarNode(key, value);
      }

      const details = document.createElement("details");
      details.open = depth === 0;

      const summary = document.createElement("summary");
      const size = Array.isArray(value) ? value.length : Object.keys(value).length;
      const open = Array.isArray(value) ? "[" : "{";
      const close = Array.isArray(value) ? "]" : "}";
      summary.textContent = `${key} ${open}${size}${close}`;
      details.appendChild(summary);

      const entries = Array.isArray(value)
        ? value.map((item, index) => [String(index), item])
        : Object.entries(value);

      for (const [childKey, childValue] of entries) {
        details.appendChild(treeNode(childKey, childValue, depth + 1));
      }
      return details;
    }

    function entryMatches(entry) {
      const level = levelFilter.value;
      const search = searchFilter.value.trim().toLowerCase();
      if (level && entry.dataset.level !== level) return false;
      if (search && !entry.dataset.search.includes(search)) return false;
      return true;
    }

    function applyFilters() {
      for (const entry of logs.querySelectorAll(".entry")) {
        entry.classList.toggle("hidden", !entryMatches(entry));
      }
    }

    function appendLog(event) {
      for (const placeholder of logs.querySelectorAll(".empty")) {
        placeholder.remove();
      }

      const entry = document.createElement("details");
      entry.className = "entry";
      entry.dataset.level = event.level || "";
      entry.dataset.search = JSON.stringify(event).toLowerCase();

      const summary = document.createElement("summary");

      const timestamp = document.createElement("span");
      timestamp.className = "timestamp";
      timestamp.textContent = event.timestamp || "";
      summary.appendChild(timestamp);

      const level = document.createElement("span");
      level.className = `level ${event.level || ""}`;
      level.textContent = event.level || "";
      summary.appendChild(level);

      const eventName = document.createElement("span");
      eventName.className = "event";
      eventName.textContent = event.event || event.message || "log";
      summary.appendChild(eventName);

      const logger = document.createElement("span");
      logger.className = "logger";
      logger.textContent = event.logger || "";
      summary.appendChild(logger);

      const meta = document.createElement("span");
      meta.className = "meta";
      const bits = [];
      if (event.trace_id && event.trace_id !== "untraced") bits.push(`trace=${event.trace_id}`);
      if (event.thread_id) bits.push(`thread=${event.thread_id}`);
      meta.textContent = bits.join(" ");
      summary.appendChild(meta);

      entry.appendChild(summary);

      const tree = document.createElement("div");
      tree.className = "tree";
      tree.appendChild(treeNode("json", event));
      entry.appendChild(tree);

      logs.appendChild(entry);

      while (logs.querySelectorAll(".entry").length > maxEntries) {
        logs.querySelector(".entry")?.remove();
      }

      entry.classList.toggle("hidden", !entryMatches(entry));

      if (autoscroll.checked) {
        window.scrollTo({top: document.body.scrollHeight, behavior: "auto"});
      }
    }

    levelFilter.addEventListener("change", applyFilters);
    searchFilter.addEventListener("input", applyFilters);

    document.getElementById("expand").addEventListener("click", () => {
      for (const details of logs.querySelectorAll("details")) details.open = true;
    });

    document.getElementById("collapse").addEventListener("click", () => {
      for (const details of logs.querySelectorAll("details")) details.open = false;
    });

    document.getElementById("clear").addEventListener("click", () => {
      logs.replaceChildren();
      const message = document.createElement("div");
      message.className = "empty";
      message.textContent = "View cleared. New events will continue to appear.";
      logs.appendChild(message);
    });

    const source = new EventSource("/logs/stream?tail=200");

    source.onopen = () => {
      status.textContent = "live";
      status.className = "connected";
    };

    source.onerror = () => {
      status.textContent = "reconnecting…";
      status.className = "error";
    };

    source.onmessage = (message) => {
      try {
        appendLog(JSON.parse(message.data));
      } catch (error) {
        console.error("Invalid log event", error, message.data);
      }
    };
  </script>
</body>
</html>
"""


def _ensure_enabled() -> None:
    if not get_settings().log_viewer_enabled:
        raise HTTPException(status_code=404, detail="Log viewer is disabled")


def _json_event(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None

    try:
        json.loads(text)
    except json.JSONDecodeError:
        return None
    return text


async def _stream_log(request: Request, tail: int) -> AsyncIterator[str]:
    path = Path(get_settings().log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("r", encoding="utf-8") as handle:
        recent = deque(handle, maxlen=tail)
        for line in recent:
            event = _json_event(line)
            if event is not None:
                yield f"data: {event}\n\n"

        while True:
            if await request.is_disconnected():
                return

            line = handle.readline()
            if line:
                event = _json_event(line)
                if event is not None:
                    yield f"data: {event}\n\n"
                continue

            await asyncio.sleep(0.25)


@router.get("/logs", response_class=HTMLResponse, include_in_schema=False)
async def logs_page() -> HTMLResponse:
    _ensure_enabled()
    return HTMLResponse(LOG_VIEWER_HTML)


@router.get("/logs/stream", include_in_schema=False)
async def logs_stream(request: Request, tail: int = 200) -> StreamingResponse:
    _ensure_enabled()
    bounded_tail = min(max(tail, 0), 2000)

    return StreamingResponse(
        _stream_log(request, bounded_tail),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
