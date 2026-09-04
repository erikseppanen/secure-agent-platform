import app.reranker as reranker


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        assert pairs == [
            ("clock skew", "less relevant chunk"),
            ("clock skew", "token validation allows clock skew"),
        ]
        assert show_progress_bar is False
        return [0.2, 0.9]


def test_rerank_sync_orders_by_cross_encoder_score(monkeypatch) -> None:
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: FakeCrossEncoder(),
    )

    candidates = [
        {
            "source": "other.md",
            "chunk_index": 0,
            "content": "less relevant chunk",
            "hybrid_score": 0.032,
        },
        {
            "source": "authentication.md",
            "chunk_index": 0,
            "content": "token validation allows clock skew",
            "hybrid_score": 0.030,
        },
    ]

    results = reranker._rerank_sync("clock skew", candidates)

    assert results[0]["source"] == "authentication.md"
    assert results[0]["reranker_score"] == 0.9
    assert results[1]["reranker_score"] == 0.2
