from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

type ToolHandlerResult = Any | Awaitable[Any]
type ToolHandler = Callable[[dict[str, Any]], ToolHandlerResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    stop_reason: str = "end_turn"
