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
   |----------------------|------------------------|
   |                      |                        |
get_system_status     get_incidents       search_documents
(simulated data)          |                        |
                         PostgreSQL          local embedding
                                                   |
                                            pgvector cosine
                                            similarity search
                                                   |
                                           document chunks
```

The agent discovers tools from the MCP server with `tools/list` and executes model-selected tools through `tools/call`.

The RAG path adds a separate ingestion pipeline:

```text
knowledge/*.md
     |
     v
chunking + overlap
     |
     v
local Sentence Transformers model
     |
     v
PostgreSQL + pgvector
     |
     v
search_documents MCP tool
     |
     v
Claude uses retrieved chunks
```

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors. Embeddings are generated locally; no OpenAI API key is required.

## Local setup

Copy the environment template:

```bash
cp .env.example .env
```

Fill in the Anthropic credentials. The embedding model configuration can normally be left at its defaults.

Install/sync dependencies:

```bash
uv sync
```

The first embedding operation downloads the configured Sentence Transformers model, then subsequent embedding work runs locally.

Start PostgreSQL with pgvector installed:

```bash
docker compose up -d postgres
```

The image is `pgvector/pgvector:pg17`. If an older RAG table exists with a different vector dimension, the application drops and recreates only the derived `document_chunks` table; the incident table is left intact.

## Ingest the sample knowledge base

```bash
uv run python -m app.ingest knowledge
```

That command reads `.md` and `.txt` files, splits them into overlapping chunks, embeds each chunk locally, and stores the chunks and vectors in PostgreSQL. Re-running ingestion replaces the stored chunks for each source file.

## Run tests and the API

```bash
uv run pytest
uv run uvicorn app.main:app --reload
```

Then call `POST /chat` from `http://127.0.0.1:8000/docs`.

Try:

```text
How much clock skew does token validation allow?
```

Expected path:

```text
Claude tool_use: search_documents
        |
        v
MCP tools/call
        |
        v
semantic_search_documents()
        |
        v
embed user's query locally
        |
        v
ORDER BY embedding <=> query_vector
        |
        v
closest document chunks
        |
        v
MCP tool result
        |
        v
Claude final answer
```

Other examples:

```text
When should an authentication deployment be rolled back?
What is the billing webhook retry schedule?
At what queue depth is the documents service considered critical?
```

The existing `get_incidents` PostgreSQL MCP tool remains available for questions such as:

```text
Show me the recent authentication incidents.
```

## Inspect the MCP server directly

```bash
uv run mcp dev app/mcp_server.py
```

Or run it over stdio:

```bash
uv run python -m app.mcp_server
```
