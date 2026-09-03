import argparse
import asyncio
from pathlib import Path

from app.rag import ingest_document


SUPPORTED_SUFFIXES = {".md", ".txt"}


def discover_documents(root: Path) -> list[Path]:
    """Find ingestible text documents under a directory."""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


async def ingest_directory(root: Path) -> None:
    documents = discover_documents(root)
    if not documents:
        raise RuntimeError(f"No .md or .txt documents found under {root}")

    total_chunks = 0
    for path in documents:
        source = str(path.relative_to(root))
        content = path.read_text(encoding="utf-8")
        chunk_count = await ingest_document(source=source, content=content)
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
