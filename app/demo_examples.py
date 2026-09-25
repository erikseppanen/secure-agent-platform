from typing import Any


CHAT_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "plain_model": {
        "summary": "1. Plain model response",
        "description": "A request that should complete without an MCP tool call.",
        "value": {
            "message": "In one sentence, explain what an enterprise operations assistant does."
        },
    },
    "system_status": {
        "summary": "2. Safe MCP tool call",
        "description": "Observe Claude request get_system_status and the application execute it through MCP.",
        "value": {
            "message": "What is the current status and latency of the authentication service?"
        },
    },
    "incidents": {
        "summary": "3. PostgreSQL tool call",
        "description": "Observe get_incidents read recent incident data from PostgreSQL.",
        "value": {
            "message": "Show me the recent authentication incidents."
        },
    },
    "rag": {
        "summary": "4. RAG retrieval and reranking",
        "description": "Observe search_documents, embeddings, hybrid retrieval, and reranking.",
        "value": {
            "message": "How much clock skew does token validation allow? Use the internal documentation."
        },
    },
    "persistence_start": {
        "summary": "5a. Persistence: start a thread",
        "description": "Send this first, then reuse the returned thread_id with the next persistence example.",
        "value": {
            "message": "My maintenance window is 02:00 UTC. Remember that for this conversation thread."
        },
    },
    "persistence_continue": {
        "summary": "5b. Persistence: continue a thread",
        "description": "Paste the thread_id returned by 5a to prove checkpointed conversation history is restored.",
        "value": {
            "message": "What maintenance window did I tell you?",
            "thread_id": "PASTE_THREAD_ID_FROM_5A_HERE",
        },
    },
    "hitl_restart": {
        "summary": "6. HITL: request a sensitive action",
        "description": "Forces a restart_service tool request so LangGraph should interrupt before execution.",
        "value": {
            "message": (
                "Call the restart_service tool for the authentication service now. "
                "Do not ask me for approval yourself; the application handles required approval."
            )
        },
    },
}


APPROVAL_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "approve": {
        "summary": "Approve the interrupted action",
        "description": "Paste the thread_id from the HITL /chat response and resume its pending approval with true.",
        "value": {
            "thread_id": "PASTE_THREAD_ID_HERE",
            "approved": True,
        },
    },
    "reject": {
        "summary": "Reject the interrupted action",
        "description": "Resume the same checkpointed thread with false so the sensitive tool is not executed.",
        "value": {
            "thread_id": "PASTE_THREAD_ID_HERE",
            "approved": False,
        },
    },
}
