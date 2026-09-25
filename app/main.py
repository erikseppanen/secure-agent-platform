from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import AgentRunResult, resume_agent, run_agent
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


class PendingApproval(BaseModel):
    details: dict[str, Any]


class ChatResponse(BaseModel):
    status: Literal["completed", "approval_required"]
    answer: str | None = None
    thread_id: str
    approval: PendingApproval | None = None


class ApprovalDecisionRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool


def _chat_response(thread_id: str, result: AgentRunResult) -> ChatResponse:
    approval = None
    if result["status"] == "approval_required":
        assert result["approval"] is not None
        approval = PendingApproval(details=result["approval"])

    return ChatResponse(
        status=result["status"],
        answer=result["answer"],
        thread_id=thread_id,
        approval=approval,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    result = await run_agent(request.message, thread_id)
    return _chat_response(thread_id, result)


@app.post("/approval", response_model=ChatResponse)
async def approval(request: ApprovalDecisionRequest) -> ChatResponse:
    result = await resume_agent(
        request.thread_id,
        request.approved,
    )
    return _chat_response(request.thread_id, result)
