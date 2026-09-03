CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS incidents (
    id BIGSERIAL PRIMARY KEY,
    service TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status TEXT NOT NULL CHECK (status IN ('open', 'monitoring', 'resolved')),
    summary TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS incidents_service_started_at_idx
    ON incidents (service, started_at DESC);

INSERT INTO incidents (service, severity, status, summary, started_at)
VALUES
    ('authentication', 'high', 'monitoring', 'Elevated login latency after identity-provider deployment.', NOW() - INTERVAL '35 minutes'),
    ('documents', 'medium', 'resolved', 'Document indexing queue accumulated backlog.', NOW() - INTERVAL '1 day'),
    ('billing', 'low', 'resolved', 'Delayed webhook delivery from payment processor.', NOW() - INTERVAL '3 days'),
    ('authentication', 'critical', 'resolved', 'Token validation failures affected a subset of requests.', NOW() - INTERVAL '8 days');

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS document_chunks_search_gin_idx
    ON document_chunks
    USING gin (search_vector);
