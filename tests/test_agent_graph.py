from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

import app.agent_graph as agent_graph_module
from app.agent_graph import AgentState, build_agent_graph, route_after_model


def _state(stop_reason: str) -> AgentState:
    return {
        "messages": [],
        "trace_id": "trace-123",
        "turn": 1,
        "stop_reason": stop_reason,
        "answer": "",
    }


def test_route_after_model_executes_tools_for_tool_use() -> None:
    assert route_after_model(_state("tool_use")) == "execute_tools"


def test_route_after_model_ends_when_claude_is_done() -> None:
    assert route_after_model(_state("end_turn")) == END


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text

    def model_dump(self) -> dict[str, str]:
        return {"type": "text", "text": self.text}


class _FakeMessages:
    def __init__(self):
        self.calls: list[list[dict]] = []

    async def create(self, *, messages, **_kwargs):
        self.calls.append(messages)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[_FakeTextBlock(f"reply-{len(self.calls)}")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessages()


@pytest.mark.asyncio
async def test_checkpointed_thread_reuses_messages_and_isolates_other_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAnthropicClient()
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
