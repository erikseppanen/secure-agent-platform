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
   |----------------------|
   |                      |
get_system_status     get_incidents
(simulated data)          |
                         PostgreSQL
```

The model does not own the tool implementations. The agent discovers tools from the MCP server with `tools/list`, translates the MCP schemas to Anthropic tool definitions, and executes model-selected tools through `tools/call`.

`get_incidents` is our first real external capability. It queries PostgreSQL through a constrained, parameterized read-only function instead of exposing arbitrary SQL to the model.

## Local setup

Copy the environment template and add your Anthropic credentials:

```bash
cp .env.example .env
```

Start PostgreSQL:

```bash
docker compose up -d postgres
```

The first startup executes `db/init.sql`, which creates and seeds the `incidents` table. The default connection string is:

```text
postgresql://sap:sap@localhost:5432/sap
```

Install/sync Python dependencies:

```bash
uv sync
```

Run the tests:

```bash
uv run pytest
```

Run the API:

```bash
uv run uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` and call `POST /chat`.

Try questions such as:

```text
What incidents have happened recently?
```

```text
Show me the recent authentication incidents.
```

The resulting path is:

```text
Claude tool_use: get_incidents
        |
        v
agent.py
        |
        v
MCP tools/call
        |
        v
mcp_server.py
        |
        v
get_recent_incidents()
        |
        v
parameterized PostgreSQL SELECT
        |
        v
MCP tool result
        |
        v
Claude final answer
```

## Inspect the MCP server directly

```bash
uv run mcp dev app/mcp_server.py
```

Or run it over stdio:

```bash
uv run python -m app.mcp_server
```
