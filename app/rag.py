from collections.abc import Mapping, Sequence
from typing import Any

import asyncpg

from app.config import get_settings
from app.embeddings import embed_text, embed_texts
from app.reranker import rerank_documents


DEFAULT_CHUNK_SIZE_WORDS = 180
DEFAULT_CHUNK_OVERLAP_WORDS = 30
DEFAULT_RRF_K = 60
HYBRID_CANDIDATE_MULTIPLIER = 4
MAX_HYBRID_CANDIDATES = 50
RERANK_CANDIDATE_MULTIPLIER = 3
MAX_RERANK_CANDIDATES = 20


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


def _normalize_metadata_value(value: str | None) -> str | None:
    """Normalize optional metadata values used for storage and filtering."""

    if value is None:
        return None

    normalized = value.strip().lower()
    return normalized or None


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
    """Enable pgvector and create vector, text-search, and metadata schema."""

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
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS service TEXT"
    )
    await connection.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS document_type TEXT"
    )
    await connection.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS environment TEXT"
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
    await connection.execute(
        "CREATE INDEX IF NOT EXISTS document_chunks_service_idx ON document_chunks (service)"
    )
    await connection.execute(
        "CREATE INDEX IF NOT EXISTS document_chunks_document_type_idx ON document_chunks (document_type)"
    )
    await connection.execute(
        "CREATE INDEX IF NOT EXISTS document_chunks_environment_idx ON document_chunks (environment)"
    )


async def ingest_document(
    source: str,
    content: str,
    service: str | None = None,
    document_type: str | None = None,
    environment: str | None = None,
) -> int:
    """Chunk, embed, and replace all stored chunks for one document."""

    chunks = chunk_text(content)
    if not chunks:
        return 0

    embeddings = await embed_texts(chunks)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding model returned an unexpected number of vectors")

    normalized_service = _normalize_metadata_value(service)
    normalized_document_type = _normalize_metadata_value(document_type)
    normalized_environment = _normalize_metadata_value(environment)

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
                INSERT INTO document_chunks (
                    source,
                    chunk_index,
                    content,
                    embedding,
                    service,
                    document_type,
                    environment
                )
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
                """,
                [
                    (
                        source,
                        index,
                        chunk,
                        _vector_literal(embedding),
                        normalized_service,
                        normalized_document_type,
                        normalized_environment,
                    )
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
    service: str | None = None,
    document_type: str | None = None,
    environment: str | None = None,
) -> list[dict[str, Any]]:
    """Return semantically similar chunks within optional metadata filters."""

    if not query.strip():
        return []

    safe_limit = max(1, min(limit, 10))
    query_vector = _vector_literal(await embed_text(query))
    service = _normalize_metadata_value(service)
    document_type = _normalize_metadata_value(document_type)
    environment = _normalize_metadata_value(environment)

    connection = await _connect()
    try:
        await ensure_rag_schema(connection)

        rows: Sequence[asyncpg.Record] = await connection.fetch(
            """
            SELECT
                source,
                chunk_index,
                content,
                service,
                document_type,
                environment,
                1 - (embedding <=> $1::vector) AS similarity
            FROM document_chunks
            WHERE ($2::text IS NULL OR service = $2)
              AND ($3::text IS NULL OR document_type = $3)
              AND ($4::text IS NULL OR environment = $4)
            ORDER BY embedding <=> $1::vector
            LIMIT $5
            """,
            query_vector,
            service,
            document_type,
            environment,
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


def _rerank_candidate_limit(limit: int) -> int:
    """Keep a broader RRF result set for the more precise reranking stage."""

    return min(
        max(limit * RERANK_CANDIDATE_MULTIPLIER, limit),
        MAX_RERANK_CANDIDATES,
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
                "service": row.get("service"),
                "document_type": row.get("document_type"),
                "environment": row.get("environment"),
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
    service: str | None = None,
    document_type: str | None = None,
    environment: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid-search, fuse candidates, and rerank the best chunks."""

    if not query.strip():
        return []

    safe_limit = max(1, min(limit, 10))
    rerank_limit = _rerank_candidate_limit(safe_limit)
    candidate_limit = _hybrid_candidate_limit(rerank_limit)
    query_vector = _vector_literal(await embed_text(query))
    service = _normalize_metadata_value(service)
    document_type = _normalize_metadata_value(document_type)
    environment = _normalize_metadata_value(environment)

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
                service,
                document_type,
                environment,
                1 - (embedding <=> $1::vector) AS vector_similarity
            FROM document_chunks
            WHERE ($2::text IS NULL OR service = $2)
              AND ($3::text IS NULL OR document_type = $3)
              AND ($4::text IS NULL OR environment = $4)
            ORDER BY embedding <=> $1::vector
            LIMIT $5
            """,
            query_vector,
            service,
            document_type,
            environment,
            candidate_limit,
        )

        keyword_rows: Sequence[asyncpg.Record] = await connection.fetch(
            """
            SELECT
                id,
                source,
                chunk_index,
                content,
                service,
                document_type,
                environment,
                ts_rank_cd(
                    search_vector,
                    websearch_to_tsquery('english', $1::text)
                ) AS keyword_score
            FROM document_chunks
            WHERE search_vector @@ websearch_to_tsquery('english', $1::text)
              AND ($2::text IS NULL OR service = $2)
              AND ($3::text IS NULL OR document_type = $3)
              AND ($4::text IS NULL OR environment = $4)
            ORDER BY keyword_score DESC
            LIMIT $5
            """,
            query,
            service,
            document_type,
            environment,
            candidate_limit,
        )

        fused_candidates = _fuse_ranked_results(
            vector_rows=vector_rows,
            keyword_rows=keyword_rows,
            limit=rerank_limit,
        )
    finally:
        await connection.close()

    return await rerank_documents(
        query=query,
        candidates=fused_candidates,
        limit=safe_limit,
    )
