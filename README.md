# Secure Agent Platform

A learning and portfolio project for building production-style agentic AI systems.

## Current architecture

```text
HTTP client
   |
FastAPI /chat
   |
Anthropic agent loop
   |
MCP client
   |
stdio transport
   |
MCP server
   |
get_system_status tool
```

The model no longer receives hand-written tool definitions from the application. The agent discovers tools from the MCP server with `tools/list`, translates the MCP schema to Anthropic's tool schema, and executes model-selected tools through `tools/call`.

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

## Inspect the MCP server directly

```bash
uv run mcp dev app/mcp_server.py
```

Or run it over stdio:

```bash
uv run python -m app.mcp_server
```

## Tests

```bash
uv run pytest
```
