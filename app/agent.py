import logging
from time import perf_counter

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.mcp_client import (
    anthropic_tool_definition,
    create_mcp_client,
    tool_result_text,
)
from app.observability import configure_logging, log_event, trace_context


configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

client = AsyncAnthropic(
    api_key=settings.anthropic_api_key,
    default_headers={
        "anthropic-workspace-id": settings.anthropic_workspace_id
    },
)


async def run_agent(user_message: str) -> str:
    with trace_context() as trace_id:
        started = perf_counter()
        log_event(
            logger,
            "agent.start",
            trace_id_created=trace_id,
            payload={"user_message": user_message},
        )

        messages = [
            {
                "role": "user",
                "content": user_message,
            }
        ]

        try:
            async with create_mcp_client() as mcp_client:
                tools_response = await mcp_client.list_tools()
                tools = [
                    anthropic_tool_definition(tool)
                    for tool in tools_response.tools
                ]
                log_event(
                    logger,
                    "mcp.tools.list",
                    tool_names=[tool["name"] for tool in tools],
                )

                turn = 0
                while True:
                    turn += 1
                    claude_started = perf_counter()
                    log_event(
                        logger,
                        "claude.request",
                        model=settings.anthropic_model,
                        turn=turn,
                        message_count=len(messages),
                        payload={
                            "messages": messages,
                            "tools": tools,
                        },
                    )

                    response = await client.messages.create(
                        model=settings.anthropic_model,
                        max_tokens=1024,
                        system=(
                            "You are an enterprise operations assistant. "
                            "Use available tools when they are needed to "
                            "answer factual questions about internal systems."
                        ),
                        tools=tools,
                        messages=messages,
                    )

                    usage = getattr(response, "usage", None)
                    log_event(
                        logger,
                        "claude.response",
                        turn=turn,
                        stop_reason=response.stop_reason,
                        duration_ms=round(
                            (perf_counter() - claude_started) * 1000,
                            1,
                        ),
                        input_tokens=getattr(usage, "input_tokens", None),
                        output_tokens=getattr(usage, "output_tokens", None),
                        payload={"content": response.content},
                    )

                    if response.stop_reason != "tool_use":
                        text_blocks = [
                            block.text
                            for block in response.content
                            if block.type == "text"
                        ]
                        answer = "\n".join(text_blocks)
                        log_event(
                            logger,
                            "agent.complete",
                            duration_ms=round(
                                (perf_counter() - started) * 1000,
                                1,
                            ),
                            payload={"answer": answer},
                        )
                        return answer

                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                        }
                    )

                    tool_results = []

                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        tool_started = perf_counter()
                        log_event(
                            logger,
                            "mcp.tool.request",
                            tool=block.name,
                            tool_use_id=block.id,
                            payload={"arguments": block.input},
                        )

                        result = await mcp_client.call_tool(
                            block.name,
                            block.input,
                        )
                        result_text = tool_result_text(result)

                        log_event(
                            logger,
                            "mcp.tool.response",
                            tool=block.name,
                            tool_use_id=block.id,
                            is_error=result.is_error,
                            duration_ms=round(
                                (perf_counter() - tool_started) * 1000,
                                1,
                            ),
                            payload={"result": result_text},
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "is_error": result.is_error,
                                "content": result_text,
                            }
                        )

                    messages.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )
        except Exception:
            logger.exception(
                '{"event":"agent.error","trace_id":"%s"}',
                trace_id,
            )
            raise
