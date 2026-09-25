# Lab 03 — Retrieval-Augmented Generation

## Goal

Follow one question through tool selection, local embeddings, PostgreSQL retrieval, reranking, and the final model answer.

## Prerequisite

Ingest the sample knowledge base if needed:

```bash
uv run python -m app.ingest knowledge
```

## Steps

1. In Swagger `POST /chat`, select **4. RAG retrieval and reranking**.
2. Click **Execute**.
3. In `/logs`, find `rag.request` and `rag.response` for this trace.
4. Expand the retrieved document chunks and compare them with the final answer.

## What to observe

RAG is implemented as a tool path. Claude asks for `search_documents`; the application executes the retrieval pipeline and returns retrieved text as a tool result.

```text
Claude tool_use(search_documents)
  → MCP search_documents
  → local query embedding
  → PostgreSQL hybrid retrieval
  → local cross-encoder reranking
  → retrieved chunks
  → tool_result
  → Claude final answer
```

The model does not query pgvector directly. The application owns the retrieval implementation.
