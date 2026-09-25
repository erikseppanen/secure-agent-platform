# Lab 04 — Persistent Conversation State

## Goal

See how `thread_id` identifies checkpointed LangGraph state across separate HTTP requests and even across process restarts.

## Steps

1. In Swagger `POST /chat`, select **5a. Persistence: start a thread** and execute it.
2. Copy the returned `thread_id`.
3. Select **5b. Persistence: continue a thread**.
4. Replace the placeholder `thread_id` with the value from step 2.
5. Execute the second request.
6. In `/logs`, compare the two requests.

## What to observe

The requests have separate trace IDs because they are separate executions, but they use the same thread ID. The PostgreSQL checkpointer restores the prior LangGraph message state before the second request continues.

```text
request 1
  trace A
  thread X
  → checkpoint saved

request 2
  trace B
  thread X
  → checkpoint X loaded
  → previous messages available
```

## Process-restart proof

1. Execute 5a and save its `thread_id`.
2. Stop Uvicorn.
3. Start Uvicorn again.
4. Execute 5b using the saved `thread_id`.

If the model can still answer from the earlier conversation, the relevant state survived because it was persisted in PostgreSQL rather than only held in Python memory.
