from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command

import app.agent_graph as agent_graph_module
from app.agent_graph import AgentState, build_agent_graph, route_after_model
from app.mcp_client import INTERNAL_TRACE_ARGUMENT


def _state(stop_reason: str, tool_name: str | None = None) -> AgentState:
    messages = []
    if tool_name is not None:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu-1",
                        "name": tool_name,
                        "input": {"service": "authentication"},
                    }
                ],
            }
        ]

    return {
        "messages": messages,
        "trace_id": "trace-123",
        "turn": 1,
        "stop_reason": stop_reason,
        "answer": "",
        "approval_granted": None,
    }


def test_route_after_model_executes_safe_tools_directly() -> None:
    assert route_after_model(_state("tool_use", "get_system_status")) == "execute_tools"


def test_route_after_model_requires_approval_for_sensitive_tools() -> None:
    assert route_after_model(_state("tool_use", "restart_service")) == "approval_gate"


def test_route_after_model_ends_when_claude_is_done() -> None:
    assert route_after_model(_state("end_turn")) == END


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text

    def model_dump(self) -> dict[str, str]:
        return {"type": "text", "text": self.text}


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name: str, arguments: dict, tool_use_id: str = "toolu-1"):
        self.name = name
        self.input = arguments
        self.id = tool_use_id

    def model_dump(self) -> dict:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


def _text_response(text: str):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[_FakeTextBlock(text)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _tool_response(name: str, arguments: dict):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[_FakeToolUseBlock(name, arguments)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


class _ScriptedMessages:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def create(self, *, messages, **_kwargs):
        self.calls.append(messages)
        return self.responses.pop(0)


class _FakeAnthropicClient:
    def __init__(self, responses: list):
        self.messages = _ScriptedMessages(responses)


class _FakeMCPClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            content=[],
            structured_content={"status": "completed"},
        )


@pytest.mark.asyncio
async def test_checkpointed_thread_reuses_messages_and_isolates_other_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAnthropicClient(
        [
            _text_response("reply-1"),
            _text_response("reply-2"),
            _text_response("reply-3"),
        ]
    )
    monkeypatch.setattr(
        agent_graph_module,
        "get_anthropic_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        agent_graph_module,
        "get_anthropic_tools",
        lambda: [],
    )

    graph = build_agent_graph(checkpointer=InMemorySaver())

    thread_a = {"configurable": {"thread_id": "thread-a"}}
    first = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "first"}],
            "trace_id": "trace-1",
            "stop_reason": None,
            "answer": "",
        },
        config=thread_a,
    )
    second = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "second"}],
            "trace_id": "trace-2",
            "stop_reason": None,
            "answer": "",
        },
        config=thread_a,
    )

    thread_b = {"configurable": {"thread_id": "thread-b"}}
    other = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "other"}],
            "trace_id": "trace-3",
            "stop_reason": None,
            "answer": "",
        },
        config=thread_b,
    )

    assert [message["content"] for message in first["messages"]] == [
        "first",
        [{"type": "text", "text": "reply-1"}],
    ]
    assert [message["content"] for message in second["messages"]] == [
        "first",
        [{"type": "text", "text": "reply-1"}],
        "second",
        [{"type": "text", "text": "reply-2"}],
    ]
    assert [message["content"] for message in other["messages"]] == [
        "other",
        [{"type": "text", "text": "reply-3"}],
    ]


@pytest.mark.asyncio
async def test_sensitive_tool_interrupts_before_execution_and_runs_after_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAnthropicClient(
        [
            _tool_response("restart_service", {"service": "authentication"}),
            _text_response("Restart completed."),
        ]
    )
    fake_mcp = _FakeMCPClient()
    monkeypatch.setattr(agent_graph_module, "get_anthropic_client", lambda: fake_client)
    monkeypatch.setattr(agent_graph_module, "get_anthropic_tools", lambda: [])
    monkeypatch.setattr(agent_graph_module, "get_mcp_client", lambda: fake_mcp)

    graph = build_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-thread"}}
    interrupted = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Restart authentication"}],
            "trace_id": "trace-before-approval",
            "stop_reason": None,
            "answer": "",
            "approval_granted": None,
        },
        config=config,
    )

    assert fake_mcp.calls == []
    interrupts = interrupted["__interrupt__"]
    assert len(interrupts) == 1
    current = interrupts[0]
    assert current.value["type"] == "tool_approval"
    assert current.value["tools"][0]["name"] == "restart_service"

    completed = await graph.ainvoke(
        Command(resume=True),
        config=config,
    )

    assert completed["answer"] == "Restart completed."
    assert len(fake_mcp.calls) == 1
    tool_name, arguments = fake_mcp.calls[0]
    assert tool_name == "restart_service"
    assert arguments["service"] == "authentication"
    assert arguments[INTERNAL_TRACE_ARGUMENT] == "trace-before-approval"


@pytest.mark.asyncio
async def test_rejected_sensitive_tool_is_not_executed_and_claude_gets_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAnthropicClient(
        [
            _tool_response("restart_service", {"service": "authentication"}),
            _text_response("Restart was not approved."),
        ]
    )
    fake_mcp = _FakeMCPClient()
    monkeypatch.setattr(agent_graph_module, "get_anthropic_client", lambda: fake_client)
    monkeypatch.setattr(agent_graph_module, "get_anthropic_tools", lambda: [])
    monkeypatch.setattr(agent_graph_module, "get_mcp_client", lambda: fake_mcp)

    graph = build_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "rejection-thread"}}
    interrupted = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Restart authentication"}],
            "trace_id": "trace-before-rejection",
            "stop_reason": None,
            "answer": "",
            "approval_granted": None,
        },
        config=config,
    )
    assert interrupted["__interrupt__"]

    completed = await graph.ainvoke(
        Command(resume=False),
        config=config,
    )

    assert fake_mcp.calls == []
    assert completed["answer"] == "Restart was not approved."
    tool_result_message = fake_client.messages.calls[-1][-1]
    tool_result = tool_result_message["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is True
    assert "approval denied" in tool_result["content"].lower()


@pytest.mark.asyncio
async def test_resume_without_pending_approval_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.agent as agent_module

    graph = build_agent_graph(checkpointer=InMemorySaver())
    monkeypatch.setattr(agent_module, "get_agent_graph", lambda: graph)

    with pytest.raises(agent_module.NoPendingApprovalError):
        await agent_module.resume_agent("thread-without-interrupt", approved=True)
