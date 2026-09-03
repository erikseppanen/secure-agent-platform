from collections.abc import Mapping, Sequence
from typing import Any

import asyncpg

from app.config import get_settings
from app.embeddings import embed_text, embed_texts


DEFAULT_CHUNK_SIZE_WORDS = 180
DEFAULT_CHUNK_OVERLAP_WORDS = 30
DEFAULT_RRF_K = 60
HYBRID_CANDIDATE_MULTIPLIER = 4
MAX_HYBRID_CANDIDATES = 50


def chunk_text(
    text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping word chunks."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be non-negative and smaller than chunk_size_words"
        )

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size_words - overlap_words

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break

    return chunks


def _vector_literal(values: list[float]) -> str:
    """Serialize a vector for PostgreSQL's vector input format."""

    return "[" + ",".join(str(value) for value in values) + "]"


async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(get_settings().database_url)


async def _drop_chunk_table_if_dimension_changed(
    connection: asyncpg.Connection,
    dimensions: int,
) -> None:
    """Recreate derived RAG data if the configured vector dimension changed."""

    existing_type = await connection.fetchval(
        """
        SELECT format_type(attribute.atttypid, attribute.atttypmod)
        FROM pg_attribute AS attribute
        JOIN pg_class AS relation
          ON relation.oid = attribute.attrelid
        JOIN pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = current_schema()
          AND relation.relname = 'document_chunks'
          AND attribute.attname = 'embedding'
          AND NOT attribute.attisdropped
        """
    )

    expected_type = f"vector({dimensions})"
    if existing_type is not None and existing_type != expected_type:
        await connection.execute("DROP TABLE document_chunks")


async def ensure_rag_schema(connection: asyncpg.Connection) -> None:
    """Enable pgvector and create the vector/full-text search schema."""

    dimensions = get_settings().embedding_dimensions
    if dimensions <= 0 or dimensions > 2000:
        raise ValueError("embedding_dimensions must be between 1 and 2000")

    await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await _drop_chunk_table_if_dimension_changed(connection, dimensions)
    await connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (source, chunk_index)
        )
        """
    )
    await connection.execute(
        """
        ALTER TABLE document_chunks
        ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )
    await connection.execute(
        """
        CREATE INDEX IF NOT EXISTS document_chunks_search_gin_idx
        ON document_chunks
        USING gin (search_vector)
        """
    )


async def ingest_document(source: str, content: str) -> int:
    """Chunk, embed, and replace all stored chunks for one document."""

    chunks = chunk_text(content)
    if not chunks:
        return 0

    embeddings = await embed_texts(chunks)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding model returned an unexpected number of vectors")

    connection = await _connect()
    try:
        await ensure_rag_schema(connection)

        async with connection.transaction():
            await connection.execute(
                "DELETE FROM document_chunks WHERE source = $1",
                source,
            )
            await connection.executemany(
                """
                INSERT INTO document_chunks (source, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4::vector)
                """,
                [
                    (source, index, chunk, _vector_literal(embedding))
                    for index, (chunk, embedding) in enumerate(
                        zip(chunks, embeddings, strict=True)
                    )
                ],
            )

        return len(chunks)
    finally:
        await connection.close()


async def semantic_search_documents(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return document chunks nearest to a natural-language query."""

    if not query.strip():
        return []

    safe_limit = max(1, min(limit, 10))
    query_vector = _vector_literal(await embed_text(query))

    connection = await _connect()
    try:
        await ensure_rag_schema(connection)

        rows: Sequence[asyncpg.Record] = await connection.fetch(
            """
            SELECT
                source,
                chunk_index,
                content,
                1 - (embedding <=> $1::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            query_vector,
            safe_limit,
        )

        return [
            {
                **dict(row),
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]
    finally:
        await connection.close()


def _hybrid_candidate_limit(limit: int) -> int:
    """Retrieve more candidates than we ultimately return before rank fusion."""

    return min(
        max(limit * HYBRID_CANDIDATE_MULTIPLIER, limit),
        MAX_HYBRID_CANDIDATES,
    )


def _fuse_ranked_results(
    vector_rows: Sequence[Mapping[str, Any]],
    keyword_rows: Sequence[Mapping[str, Any]],
    limit: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Combine vector and keyword rankings with Reciprocal Rank Fusion."""

    combined: dict[int, dict[str, Any]] = {}

    def get_result(row: Mapping[str, Any]) -> dict[str, Any]:
        row_id = int(row["id"])
        if row_id not in combined:
            combined[row_id] = {
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "vector_similarity": None,
                "keyword_score": None,
                "hybrid_score": 0.0,
            }
        return combined[row_id]

    for rank, row in enumerate(vector_rows, start=1):
        result = get_result(row)
        result["vector_similarity"] = float(row["vector_similarity"])
        result["hybrid_score"] += 1.0 / (rrf_k + rank)

    for rank, row in enumerate(keyword_rows, start=1):
        result = get_result(row)
        result["keyword_score"] = float(row["keyword_score"])
        result["hybrid_score"] += 1.0 / (rrf_k + rank)

    def sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
        vector_similarity = result["vector_similarity"]
        keyword_score = result["keyword_score"]
        return (
            float(result["hybrid_score"]),
            -1.0 if vector_similarity is None else float(vector_similarity),
            -1.0 if keyword_score is None else float(keyword_score),
        )

    ranked = sorted(combined.values(), key=sort_key, reverse=True)
    return ranked[:limit]


async def hybrid_search_documents(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Combine semantic and PostgreSQL full-text search for document retrieval."""

    if not query.strip():
        return []

    safe_limit = max(1, min(limit, 10))
    candidate_limit = _hybrid_candidate_limit(safe_limit)
    query_vector = _vector_literal(await embed_text(query))

    connection = await _connect()
    try:
        await ensure_rag_schema(connection)

        vector_rows: Sequence[asyncpg.Record] = await connection.fetch(
            """
            SELECT
                id,
                source,
                chunk_index,
                content,
                1 - (embedding <=> $1::vector) AS vector_similarity
            FROM document_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            query_vector,
            candidate_limit,
        )

        keyword_rows: Sequence[asyncpg.Record] = await connection.fetch(
            """
            SELECT
                id,
                source,
                chunk_index,
                content,
                ts_rank_cd(
                    search_vector,
                    websearch_to_tsquery('english', $1::text)
                ) AS keyword_score
            FROM document_chunks
            WHERE search_vector @@ websearch_to_tsquery('english', $1::text)
            ORDER BY keyword_score DESC
            LIMIT $2
            """,
            query,
            candidate_limit,
        )

        return _fuse_ranked_results(
            vector_rows=vector_rows,
            keyword_rows=keyword_rows,
            limit=safe_limit,
        )
    finally:
        await connection.close()
