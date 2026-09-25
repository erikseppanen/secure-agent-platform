# Secure Agent Platform Learning Labs

These labs are designed to be run while you keep four views open at the same time:

1. **Terminal** — start and stop the actual FastAPI process yourself.
2. **Swagger** — `http://127.0.0.1:8000/docs` for the HTTP request/response.
3. **Live logs** — `http://127.0.0.1:8000/logs` for structured JSON execution events.
4. **Learning guide** — `http://127.0.0.1:8000/learn` or these Markdown files.

Start the application explicitly:

```bash
uv run uvicorn app.main:app --reload
```

The app does not auto-run requests for these labs. In Swagger, open `POST /chat`, click **Try it out**, and use the named Example dropdown. The example numbers match the labs.

## Labs

- [01 — Plain model request](01-basic-agent.md)
- [02 — Safe MCP tool call](02-tool-calling.md)
- [03 — Retrieval-Augmented Generation](03-rag.md)
- [04 — Persistent conversation state](04-persistence.md)
- [05 — Human-in-the-loop approval](05-human-in-the-loop.md)

The `scripts/open-lab` helper only opens `/learn`, `/docs`, and `/logs` in browser tabs. It does not start the server and does not send requests.
