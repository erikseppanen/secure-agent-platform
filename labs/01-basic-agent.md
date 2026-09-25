# Lab 01 — Plain Model Request

## Goal

See the shortest possible agent path: HTTP request → LangGraph → Claude → HTTP response, with no MCP tool execution.

## Steps

1. Start the app with `uv run uvicorn app.main:app --reload`.
2. Open Swagger at `http://127.0.0.1:8000/docs`.
3. Open the live logs at `http://127.0.0.1:8000/logs`.
4. In Swagger, open `POST /chat`, click **Try it out**, and select **1. Plain model response**.
5. Click **Execute**.
6. Copy the returned `thread_id` or use the request's trace ID from the logs to filter `/logs`.

## What to observe

The model returns normal text instead of a `tool_use` block. LangGraph therefore routes directly to END.

```text
POST /chat
  → run_agent
  → LangGraph call_model
  → Claude normal response
  → route_after_model
  → END
  → HTTP response
```

There should be no MCP tool request for this lab.
