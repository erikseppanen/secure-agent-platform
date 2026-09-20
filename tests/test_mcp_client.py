from app.mcp_client import INTERNAL_TRACE_ARGUMENT, _public_input_schema


def test_public_input_schema_hides_internal_trace_argument() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            INTERNAL_TRACE_ARGUMENT: {"type": ["string", "null"]},
        },
        "required": ["query", INTERNAL_TRACE_ARGUMENT],
    }

    public = _public_input_schema(schema)

    assert INTERNAL_TRACE_ARGUMENT not in public["properties"]
    assert public["required"] == ["query"]
    assert INTERNAL_TRACE_ARGUMENT in schema["properties"]
