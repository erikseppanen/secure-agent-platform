import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the local embedding model."""

    settings = get_settings()
    model = SentenceTransformer(
        settings.embedding_model,
        local_files_only=True)

    dimensions = model.get_embedding_dimension()
    if dimensions != settings.embedding_dimensions:
        raise RuntimeError(
            "Embedding model dimension does not match EMBEDDING_DIMENSIONS: "
            f"model={dimensions}, configured={settings.embedding_dimensions}"
        )

    return model


def _encode(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text locally without blocking the async event loop."""

    if not texts:
        return []

    return await asyncio.to_thread(_encode, texts)


async def embed_text(text: str) -> list[float]:
    """Embed one piece of text locally."""

    return (await embed_texts([text]))[0]
