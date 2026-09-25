from fastapi.testclient import TestClient

from app.main import app


def test_learning_page_is_registered() -> None:
    assert TestClient(app).get("/learn").status_code == 200


def test_chat_swagger_examples_are_exposed() -> None:
    schema = app.openapi()
    examples = schema["paths"]["/chat"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]

    assert {
        "plain_model",
        "system_status",
        "incidents",
        "rag",
        "persistence_start",
        "persistence_continue",
        "hitl_restart",
    } <= examples.keys()


def test_approval_swagger_examples_use_thread_id_only() -> None:
    schema = app.openapi()
    examples = schema["paths"]["/approval"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]

    assert examples["approve"]["value"]["approved"] is True
    assert examples["reject"]["value"]["approved"] is False
    assert "interrupt_id" not in examples["approve"]["value"]
    assert "interrupt_id" not in examples["reject"]["value"]

    request_schema = schema["components"]["schemas"]["ApprovalDecisionRequest"]
    assert "interrupt_id" not in request_schema["properties"]

    pending_schema = schema["components"]["schemas"]["PendingApproval"]
    assert "interrupt_id" not in pending_schema["properties"]
