from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter()

LEARNING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Secure Agent Platform Learning Lab</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      margin: 0;
      background: #0b1220;
      color: #e5e7eb;
      line-height: 1.5;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.65rem;
      padding: 0.8rem 1.1rem;
      background: #0f172a;
      border-bottom: 1px solid #334155;
    }
    header h1 {
      font-size: 1rem;
      margin: 0 auto 0 0;
    }
    a.button {
      display: inline-block;
      text-decoration: none;
      color: #e5e7eb;
      border: 1px solid #475569;
      border-radius: 0.4rem;
      padding: 0.35rem 0.65rem;
      background: #111827;
    }
    a.button:hover { background: #1e293b; }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 1.25rem;
    }
    .intro, .lab {
      border: 1px solid #334155;
      border-radius: 0.65rem;
      background: #0f172a;
      margin-bottom: 1rem;
      padding: 1rem 1.1rem;
    }
    .intro h2, .lab h2 { margin-top: 0; }
    .surface-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.7rem;
      margin: 0.8rem 0;
    }
    .surface {
      border: 1px solid #334155;
      border-radius: 0.5rem;
      padding: 0.7rem;
      background: #111827;
    }
    .surface strong { color: #93c5fd; }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    code {
      color: #c4b5fd;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid #334155;
      border-radius: 0.45rem;
      background: #020617;
      padding: 0.7rem;
      color: #d1fae5;
    }
    details {
      border-top: 1px solid #334155;
      padding-top: 0.7rem;
      margin-top: 0.7rem;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
      color: #bfdbfe;
    }
    ol li, ul li { margin-bottom: 0.45rem; }
    .watch {
      border-left: 3px solid #22d3ee;
      padding-left: 0.75rem;
      color: #cbd5e1;
    }
    .important {
      border-left: 3px solid #f59e0b;
      padding-left: 0.75rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Secure Agent Platform — Learning Lab</h1>
    <a class="button" href="/docs" target="_blank">Swagger</a>
    <a class="button" href="/logs" target="_blank">Live Logs</a>
  </header>
  <main>
    <section class="intro">
      <h2>How to use this lab</h2>
      <p>Start the application yourself in a terminal, then keep Swagger and the live log viewer open beside this page. The lab does not start the server or send requests for you.</p>
      <pre>uv run uvicorn app.main:app --reload</pre>
      <div class="surface-grid">
        <div class="surface"><strong>Terminal</strong><br>Process lifecycle and normal application logs.</div>
        <div class="surface"><strong>Swagger /docs</strong><br>The actual HTTP requests and responses you control.</div>
        <div class="surface"><strong>Live Logs /logs</strong><br>Structured JSON events, trace IDs, tool calls, and payloads.</div>
        <div class="surface"><strong>This page /learn</strong><br>What to do, what to inspect, and why it matters.</div>
      </div>
      <p class="watch">In Swagger, open <code>POST /chat</code>, click <strong>Try it out</strong>, and use the named Example dropdown. The example numbers below match the Swagger examples.</p>
    </section>

    <section class="lab">
      <h2>Lab 1 — Plain model request</h2>
      <ol>
        <li>In Swagger <code>POST /chat</code>, select <strong>1. Plain model response</strong>.</li>
        <li>Click <strong>Execute</strong>.</li>
        <li>In <code>/logs</code>, filter using the returned trace ID if you want to isolate this request.</li>
      </ol>
      <p class="watch"><strong>Notice:</strong> the model returns normal text rather than a <code>tool_use</code> block, so LangGraph reaches END without MCP execution.</p>
      <details>
        <summary>Expected control flow</summary>
        <pre>POST /chat
  → run_agent
  → LangGraph call_model
  → Claude response without tool_use
  → route_after_model
  → END
  → HTTP completed response</pre>
      </details>
    </section>

    <section class="lab">
      <h2>Lab 2 — Safe MCP tool call</h2>
      <ol>
        <li>Select <strong>2. Safe MCP tool call</strong> in <code>POST /chat</code>.</li>
        <li>Execute it and watch the JSON events in <code>/logs</code>.</li>
        <li>Expand the model response, tool request, MCP server request/response, and final model response.</li>
      </ol>
      <p class="watch"><strong>Notice:</strong> Claude does not execute the tool. It emits a tool request. The application routes and executes the MCP tool, appends the result, and calls Claude again.</p>
      <details>
        <summary>Expected control flow</summary>
        <pre>Claude tool_use(get_system_status)
  → route_after_model
  → execute_tools
  → MCP client tools/call
  → MCP server get_system_status
  → tool_result appended to graph state
  → Claude final answer</pre>
      </details>
    </section>

    <section class="lab">
      <h2>Lab 3 — Retrieval-Augmented Generation</h2>
      <ol>
        <li>Make sure the sample knowledge base has been ingested.</li>
        <li>Select <strong>4. RAG retrieval and reranking</strong> in <code>POST /chat</code>.</li>
        <li>Execute it and look for <code>rag.request</code> and <code>rag.response</code> in <code>/logs</code>.</li>
        <li>Expand the retrieved results and compare them with the final answer.</li>
      </ol>
      <p class="watch"><strong>Notice:</strong> RAG is a tool path. The model requests <code>search_documents</code>; the application executes retrieval and reranking; retrieved text comes back as a tool result that the model uses as context.</p>
      <details>
        <summary>Expected control flow</summary>
        <pre>Claude tool_use(search_documents)
  → MCP search_documents
  → local query embedding
  → PostgreSQL hybrid retrieval
  → local cross-encoder reranking
  → retrieved chunks
  → tool_result
  → Claude final answer</pre>
      </details>
    </section>

    <section class="lab">
      <h2>Lab 4 — Persistent conversation state</h2>
      <ol>
        <li>Select <strong>5a. Persistence: start a thread</strong> and execute it.</li>
        <li>Copy the returned <code>thread_id</code>.</li>
        <li>Select <strong>5b. Persistence: continue a thread</strong>.</li>
        <li>Replace the placeholder <code>thread_id</code> with the ID from step 2 and execute.</li>
      </ol>
      <p class="watch"><strong>Notice:</strong> the two HTTP requests have separate trace IDs but the same thread ID. LangGraph's PostgreSQL checkpointer restores the thread's previous message state before continuing.</p>
      <details>
        <summary>Optional restart proof</summary>
        <ol>
          <li>Complete 5a and save the returned thread ID.</li>
          <li>Stop Uvicorn.</li>
          <li>Start Uvicorn again using the same command.</li>
          <li>Execute 5b using the saved thread ID.</li>
        </ol>
        <p>If the conversation continues, the relevant state survived process restart because it was checkpointed in PostgreSQL.</p>
      </details>
    </section>

    <section class="lab">
      <h2>Lab 5 — Human-in-the-loop approval</h2>
      <p>This is the most useful multi-step lab because the workflow intentionally pauses and later resumes from a persisted checkpoint.</p>
      <h3>Step 1 — Request a sensitive action</h3>
      <ol>
        <li>Select <strong>6. HITL: request a sensitive action</strong> in <code>POST /chat</code>.</li>
        <li>Execute it.</li>
        <li>The expected response status is <code>approval_required</code>. Copy the returned <code>thread_id</code>.</li>
        <li>Inspect <code>approval.details</code> to see the sensitive tool and arguments waiting for approval.</li>
        <li>In <code>/logs</code>, find and expand <code>agent.approval.required</code>.</li>
      </ol>
      <p class="important"><strong>Important:</strong> at this point <code>restart_service</code> should not have executed. The graph was interrupted before the sensitive tool node was allowed to perform the action.</p>

      <h3>Step 2 — Resume with a human decision</h3>
      <ol>
        <li>Open <code>POST /approval</code> in Swagger.</li>
        <li>Select either <strong>Approve the interrupted action</strong> or <strong>Reject the interrupted action</strong>.</li>
        <li>Paste the saved <code>thread_id</code>.</li>
        <li>Execute the request and watch the resumed flow in <code>/logs</code>.</li>
      </ol>
      <p class="watch"><strong>Notice:</strong> the thread ID is the persistent cursor. The approval request resumes the single pending interrupt for that thread. Approval lets the sensitive MCP tool execute; rejection produces a synthetic tool error instead.</p>

      <h3>Step 3 — Prove interrupt persistence</h3>
      <ol>
        <li>Repeat Step 1 until you have a new <code>approval_required</code> response.</li>
        <li>Save the returned <code>thread_id</code>.</li>
        <li>Stop Uvicorn completely.</li>
        <li>Start it again.</li>
        <li>Use <code>POST /approval</code> with the saved thread ID.</li>
      </ol>
      <p class="watch"><strong>Notice:</strong> successful resume after process restart demonstrates that the interrupted graph execution was persisted rather than kept only in Python memory.</p>
      <details>
        <summary>Expected control flow</summary>
        <pre>POST /chat
  → call_model
  → Claude tool_use(restart_service)
  → route_after_model
  → approval_gate
  → interrupt()
  → checkpoint persisted
  → approval_required HTTP response

POST /approval
  → Command(resume=true/false)
  → checkpoint restored by thread_id
  → approval_gate resumes
  → execute_tools
  → restart_service if approved
  → Claude final answer</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""


@router.get("/learn", response_class=HTMLResponse, include_in_schema=False)
async def learning_page() -> HTMLResponse:
    return HTMLResponse(LEARNING_HTML)
