# Lab 02 — Safe MCP Tool Call

## Goal

See the distinction between the model requesting a tool and the application actually executing it.

## Steps

1. In Swagger `POST /chat`, select **2. Safe MCP tool call**.
2. Click **Execute**.
3. In `/logs`, expand the model response that contains `tool_use`.
4. Find the MCP client/server request and response events.
5. Expand the final model response after the tool result is appended to graph state.

## What to observe

Claude does not call MCP directly. It emits a `tool_use` request. The application routes that request to the tool execution node, calls MCP, appends a `tool_result`, and invokes Claude again.

```text
Claude tool_use(get_system_status)
  → route_after_model
  → execute_tools
  → MCP client tools/call
  → MCP server get_system_status
  → tool_result appended
  → Claude final answer
```

This is the core agent loop used by the later RAG and HITL labs.
