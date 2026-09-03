import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.rag import ingest_document


SUPPORTED_SUFFIXES = {".md", ".txt"}
METADATA_FILENAME = "metadata.json"


def discover_documents(root: Path) -> list[Path]:
    """Find ingestible text documents under a directory."""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_metadata(root: Path) -> dict[str, dict[str, Any]]:
    """Load optional per-document metadata keyed by relative path."""

    metadata_path = root / METADATA_FILENAME
    if not metadata_path.exists():
        return {}

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")

    metadata: dict[str, dict[str, Any]] = {}
    for source, values in raw.items():
        if not isinstance(source, str) or not isinstance(values, dict):
            raise ValueError(
                f"{metadata_path} entries must map document paths to JSON objects"
            )
        metadata[source] = values

    return metadata


async def ingest_directory(root: Path) -> None:
    documents = discover_documents(root)
    if not documents:
        raise RuntimeError(f"No .md or .txt documents found under {root}")

    metadata_by_source = load_metadata(root)

    total_chunks = 0
    for path in documents:
        source = str(path.relative_to(root))
        content = path.read_text(encoding="utf-8")
        metadata = metadata_by_source.get(source, {})

        chunk_count = await ingest_document(
            source=source,
            content=content,
            service=metadata.get("service"),
            document_type=metadata.get("document_type"),
            environment=metadata.get("environment"),
        )
        total_chunks += chunk_count
        print(f"{source}: {chunk_count} chunks")

    print(f"Ingested {len(documents)} documents / {total_chunks} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk, embed, and store knowledge documents in pgvector."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="knowledge",
        help="Directory containing .md or .txt files (default: knowledge)",
    )
    args = parser.parse_args()

    asyncio.run(ingest_directory(Path(args.directory)))


if __name__ == "__main__":
    main()
