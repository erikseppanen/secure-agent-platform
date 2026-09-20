from types import SimpleNamespace

from langgraph.graph import END

from app.agent_graph import AgentState, route_after_model


def _state(stop_reason: str) -> AgentState:
    return {
        "messages": [],
        "trace_id": "trace-123",
        "turn": 1,
        "response": SimpleNamespace(stop_reason=stop_reason),
        "answer": "",
    }


def test_route_after_model_executes_tools_for_tool_use() -> None:
    assert route_after_model(_state("tool_use")) == "execute_tools"


def test_route_after_model_ends_when_claude_is_done() -> None:
    assert route_after_model(_state("end_turn")) == END
