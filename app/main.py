from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import run_agent
from app.mcp_runtime import start_mcp_runtime, stop_mcp_runtime


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Own application-scoped resources such as the persistent MCP subprocess."""

    await start_mcp_runtime()
    try:
        yield
    finally:
        await stop_mcp_runtime()


app = FastAPI(
    title="Secure Agent Platform",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    answer = await run_agent(request.message)

    return ChatResponse(
        answer=answer,
    )
