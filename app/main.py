from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.checkpoint_runtime import (
    start_checkpoint_runtime,
    stop_checkpoint_runtime,
)
from app.mcp_runtime import start_mcp_runtime, stop_mcp_runtime


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Own the persistent MCP subprocess and LangGraph checkpointer."""

    await start_mcp_runtime()
    try:
        await start_checkpoint_runtime()
        try:
            yield
        finally:
            await stop_checkpoint_runtime()
    finally:
        await stop_mcp_runtime()


app = FastAPI(
    title="Secure Agent Platform",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    answer = await run_agent(request.message, thread_id)

    return ChatResponse(
        answer=answer,
        thread_id=thread_id,
    )
