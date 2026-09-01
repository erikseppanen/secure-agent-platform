# Secure Agent Platform

The application keeps its FastAPI and tool-orchestration layers independent of
the model billing/authentication path. Select a provider with `LLM_PROVIDER`.

## Anthropic API provider (pay per token)

```dotenv
LLM_PROVIDER=anthropic_api
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_WORKSPACE_ID=wrkspc_...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

`ANTHROPIC_WORKSPACE_ID` is optional for keys that are not workspace scoped.

## Claude Agent SDK provider (Claude-plan Agent SDK credit)

First sign in locally with Claude Code, then select the SDK provider:

```shell
claude auth login
```

```dotenv
LLM_PROVIDER=claude_agent_sdk
CLAUDE_AGENT_MODEL=claude-sonnet-4-6
CLAUDE_AGENT_MAX_TURNS=8
CLAUDE_AGENT_USE_SUBSCRIPTION=true
```

When subscription mode is enabled, the adapter clears inherited Anthropic API
credentials before launching the SDK subprocess. This prevents an exported API
key from silently changing the request to pay-per-token billing. The local
Claude login must have Agent SDK credit available.

Subscription credentials are intended for the plan owner's ordinary, local
use. Use the API provider for a hosted product or for requests made on behalf
of other users.

## Run

```shell
uv sync
uv run uvicorn app.main:app --reload
```

Switching providers requires only changing `LLM_PROVIDER`; the FastAPI route,
agent loop, and tool implementations stay unchanged.
