import asyncio
from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import get_settings


@lru_cache
def get_reranker_model() -> CrossEncoder:
    """Load and cache the local cross-encoder reranker."""

    return CrossEncoder(
        get_settings().reranker_model,
        local_files_only=True,
    )


def _rerank_sync(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    model = get_reranker_model()
    pairs = [(query, candidate["content"]) for candidate in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    reranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        reranked.append(
            {
                **candidate,
                "reranker_score": float(score),
            }
        )

    return sorted(
        reranked,
        key=lambda result: result["reranker_score"],
        reverse=True,
    )


async def rerank_documents(
    query: str,
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Rerank retrieved chunks with a local cross-encoder."""

    if not candidates:
        return []

    safe_limit = max(1, min(limit, len(candidates)))
    reranked = await asyncio.to_thread(_rerank_sync, query, candidates)
    return reranked[:safe_limit]
