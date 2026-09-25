from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import (
    AgentRunResult,
    NoPendingApprovalError,
    resume_agent,
    run_agent,
)
from app.checkpoint_runtime import (
    start_checkpoint_runtime,
    stop_checkpoint_runtime,
)
from app.demo_examples import APPROVAL_OPENAPI_EXAMPLES, CHAT_OPENAPI_EXAMPLES
from app.learning import router as learning_router
from app.log_viewer import router as log_viewer_router
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
app.include_router(log_viewer_router)
app.include_router(learning_router)


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
async def chat(
    request: Annotated[
        ChatRequest,
        Body(openapi_examples=CHAT_OPENAPI_EXAMPLES),
    ],
) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    result = await run_agent(request.message, thread_id)
    return _chat_response(thread_id, result)


@app.post("/approval", response_model=ChatResponse)
async def approval(
    request: Annotated[
        ApprovalDecisionRequest,
        Body(openapi_examples=APPROVAL_OPENAPI_EXAMPLES),
    ],
) -> ChatResponse:
    try:
        result = await resume_agent(
            request.thread_id,
            request.approved,
        )
    except NoPendingApprovalError:
        raise HTTPException(
            status_code=409,
            detail="No approval is pending for this thread.",
        ) from None
    return _chat_response(request.thread_id, result)
