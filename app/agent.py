import json

from anthropic import Anthropic

from app.config import get_settings
from app.tools import TOOL_DEFINITIONS, execute_tool


settings = get_settings()

client = Anthropic(
    api_key=settings.anthropic_api_key,
    default_headers={
        "anthropic-workspace-id": settings.anthropic_workspace_id
    },
)

def run_agent(user_message: str) -> str:
    # starts with user's message
    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    # infinite loop
    while True:

        # call Claude
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=(
                "You are an enterprise operations assistant. "
                "Use available tools when they are needed to "
                "answer factual questions about internal systems."
            ),
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Look at Claude's response:
        # Did Claude finish answering, or did Claude stop because it wants me to perform a tool call?
        # if it simply finished answering, call this block and return
        if response.stop_reason != "tool_use":
            # python list comprehension:
            text_blocks = [
                # 3. from each matching block, extract:
                block.text
                # 1. look at every block in response.content:
                for block in response.content
                # 2. keep only blocks where:
                if block.type == "text"
            ]   # 4. and put all those strings into a list called text_blocks (at top)

            return "\n".join(text_blocks)

        # otherwisee, append Claude's response to the messages
        # (the message is that Claude wants to invoke a tool)
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        tool_results = []

        for block in response.content:
            # if we're not calling a tool, bail
            if block.type != "tool_use":
                continue

            # otherwise, try calling the tool
            try:
                # block.name  "get_system_status"
                # block.input {"service": "authentication"}
                result = execute_tool(
                    block.name,
                    block.input,
                )

                tool_result = {
                    "type": "tool_result",
                    # block.id connects the result to Claude's original request
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    # json.dumps converts a python dictionary to JSON text
                }

            # if tool execution fails, return this
            except Exception as exc:
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": str(exc),
                }

            tool_results.append(tool_result)

        # append the tool results to messages
        messages.append(
            {
                "role": "user",
                "content": tool_results,
            }

        # reaches bottom of loop, go back up to (while True)
        # (it will call Claude again, but Claude will have all the messages, including tool results)
        )