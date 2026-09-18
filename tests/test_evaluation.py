import json
from pathlib import Path

import pytest

from app.evaluate import (
    RetrievalEvalCase,
    first_relevant_rank,
    load_evaluation_cases,
    recall_at_k,
    reciprocal_rank,
)


def _result(source: str, chunk_index: int) -> dict[str, object]:
    return {
        "source": source,
        "chunk_index": chunk_index,
        "content": "example",
    }


def test_recall_at_k_finds_relevant_chunk() -> None:
    results = [
        _result("wrong.md", 0),
        _result("right.md", 2),
        _result("other.md", 0),
    ]
    relevant = frozenset({("right.md", 2)})

    assert recall_at_k(results, relevant, 1) == 0.0
    assert recall_at_k(results, relevant, 2) == 1.0


def test_recall_at_k_supports_multiple_relevant_chunks() -> None:
    results = [
        _result("doc.md", 0),
        _result("wrong.md", 0),
        _result("doc.md", 1),
    ]
    relevant = frozenset({("doc.md", 0), ("doc.md", 1)})

    assert recall_at_k(results, relevant, 1) == 0.5
    assert recall_at_k(results, relevant, 3) == 1.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    results = [
        _result("wrong.md", 0),
        _result("right.md", 0),
        _result("right.md", 1),
    ]
    relevant = frozenset({("right.md", 0), ("right.md", 1)})

    assert reciprocal_rank(results, relevant) == 0.5
    assert first_relevant_rank(results, relevant) == 2


def test_reciprocal_rank_is_zero_for_miss() -> None:
    results = [_result("wrong.md", 0)]
    relevant = frozenset({("right.md", 0)})

    assert reciprocal_rank(results, relevant) == 0.0
    assert first_relevant_rank(results, relevant) is None


def test_load_evaluation_cases_normalizes_filters(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "example",
            "query": "Where is the runbook?",
            "filters": {"service": " Authentication "},
            "relevant_chunks": [
                {"source": "authentication-runbook.md", "chunk_index": 0}
            ],
            "expected_answer": "In the authentication runbook.",
        }
    ]
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    cases = load_evaluation_cases(path)

    assert cases == [
        RetrievalEvalCase(
            case_id="example",
            query="Where is the runbook?",
            relevant_chunks=frozenset({("authentication-runbook.md", 0)}),
            filters={"service": "authentication"},
            expected_answer="In the authentication runbook.",
        )
    ]


def test_load_evaluation_cases_rejects_unknown_filter(tmp_path: Path) -> None:
    dataset = [
        {
            "id": "bad-filter",
            "query": "question",
            "filters": {"owner": "security"},
            "relevant_chunks": [
                {"source": "authentication-runbook.md", "chunk_index": 0}
            ],
        }
    ]
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported filters"):
        load_evaluation_cases(path)
