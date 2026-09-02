# Secure Agent Platform

A learning and portfolio project for building production-style agentic AI systems.

This branch (`tool-calling-agent-loop`) is the first milestone: a FastAPI service
that runs a hand-written Anthropic tool-calling loop against a small set of
locally defined tools.

## Current architecture

```text
HTTP client
   |
FastAPI /chat
   |
Anthropic agent loop  (app/agent.py)
   |
execute_tool dispatch  (app/tools.py)
   |
get_system_status tool
```

The application owns the whole loop:

1. Send the user message plus `TOOL_DEFINITIONS` to the Messages API.
2. If `stop_reason != "tool_use"`, collect the text blocks and return.
3. Otherwise append the assistant turn, run each requested tool through
   `execute_tool`, append the `tool_result` blocks as a user turn, and loop.

Tool schemas are written by hand in `app/tools.py` and dispatched by name. A
later branch (`mcp-server`) replaces this with MCP-based tool discovery.

## Configuration

Settings load from `.env` (see `app/config.py`):

| Variable                 | Required | Default             |
| ------------------------ | -------- | ------------------- |
| `ANTHROPIC_API_KEY`      | yes      | -                   |
| `ANTHROPIC_WORKSPACE_ID` | yes      | -                   |
| `ANTHROPIC_MODEL`        | no       | `claude-sonnet-4-6` |

## Run the API

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` and call `POST /chat`.

Example request:

```json
{
  "message": "What is the status of authentication?"
}
```

`GET /health` returns `{"status": "ok"}`.

## Tools

`get_system_status(service)` returns simulated status/latency for `billing`,
`authentication`, or `documents`, and an `unknown` result for anything else.
