# Lab 05 — Human-in-the-Loop Approval

## Goal

Observe a sensitive tool request pause inside LangGraph, return control to the human, and later resume from the persisted checkpoint.

## Step 1 — Request the sensitive action

1. In Swagger `POST /chat`, select **6. HITL: request a sensitive action**.
2. Click **Execute**.
3. The expected response status is `approval_required`.
4. Copy the returned `thread_id`.
5. Inspect `approval.details` to see the sensitive tool and arguments waiting for approval.
6. In `/logs`, find and expand `agent.approval.required`.

At this point `restart_service` should **not** have executed. LangGraph interrupted before the sensitive action was allowed to run.

```text
POST /chat
  → call_model
  → Claude tool_use(restart_service)
  → route_after_model
  → approval_gate
  → interrupt()
  → checkpoint persisted
  → approval_required response
```

## Step 2 — Make the human decision

1. Open Swagger `POST /approval`.
2. Select **Approve the interrupted action** or **Reject the interrupted action**.
3. Paste the `thread_id` from Step 1.
4. Leave `approved` as `true` to approve or `false` to reject.
5. Click **Execute**.
6. Watch the resumed execution in `/logs`.

Approval allows the simulated `restart_service` MCP tool to execute. Rejection skips that sensitive execution and feeds a synthetic tool error back into the graph instead.

The thread ID is the persistent cursor that tells LangGraph which checkpointed workflow to resume. The approval HTTP request is a new execution, but it resumes the single pending interrupt for the same thread.

```text
POST /approval
  → Command(resume=true/false)
  → checkpoint restored by thread_id
  → approval_gate resumes
  → execute_tools
  → restart_service if approved
  → Claude final answer
```

## Step 3 — Prove interrupted-state persistence

1. Repeat Step 1 until you have a fresh `approval_required` response.
2. Save the returned `thread_id`.
3. Stop Uvicorn completely.
4. Start it again with `uv run uvicorn app.main:app --reload`.
5. Use `POST /approval` with the saved thread ID.

A successful resume demonstrates that the paused graph execution was checkpointed in PostgreSQL rather than being held only in the original Python process.
