import pytest

import app.rag as rag


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(12))

    chunks = rag.chunk_text(text, chunk_size_words=5, overlap_words=2)

    assert chunks == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]


def test_vector_literal() -> None:
    assert rag._vector_literal([0.1, 0.2, -0.3]) == "[0.1,0.2,-0.3]"


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        rag.chunk_text("one two three", chunk_size_words=3, overlap_words=3)


def test_hybrid_candidate_limit_expands_search_pool() -> None:
    assert rag._hybrid_candidate_limit(5) == 20


def test_normalize_metadata_value() -> None:
    assert rag._normalize_metadata_value(" Authentication ") == "authentication"
    assert rag._normalize_metadata_value("   ") is None
    assert rag._normalize_metadata_value(None) is None


def test_rrf_rewards_results_found_by_both_searches() -> None:
    vector_rows = [
        {
            "id": 1,
            "source": "authentication.md",
            "chunk_index": 0,
            "content": "token validation and clock skew",
            "service": "authentication",
            "document_type": "runbook",
            "environment": "production",
            "vector_similarity": 0.90,
        },
        {
            "id": 2,
            "source": "billing.md",
            "chunk_index": 0,
            "content": "payment webhook retries",
            "service": "billing",
            "document_type": "runbook",
            "environment": "production",
            "vector_similarity": 0.80,
        },
    ]
    keyword_rows = [
        {
            "id": 3,
            "source": "documents.md",
            "chunk_index": 0,
            "content": "token validation reference",
            "service": "documents",
            "document_type": "runbook",
            "environment": "production",
            "keyword_score": 1.2,
        },
        {
            "id": 1,
            "source": "authentication.md",
            "chunk_index": 0,
            "content": "token validation and clock skew",
            "service": "authentication",
            "document_type": "runbook",
            "environment": "production",
            "keyword_score": 0.9,
        },
    ]

    results = rag._fuse_ranked_results(vector_rows, keyword_rows, limit=3)

    assert results[0]["source"] == "authentication.md"
    assert results[0]["service"] == "authentication"
    assert results[0]["document_type"] == "runbook"
    assert results[0]["environment"] == "production"
    assert results[0]["vector_similarity"] == pytest.approx(0.90)
    assert results[0]["keyword_score"] == pytest.approx(0.9)
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]
