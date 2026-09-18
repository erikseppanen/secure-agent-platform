import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag import hybrid_search_documents, semantic_search_documents


ChunkKey = tuple[str, int]
Retriever = Callable[["RetrievalEvalCase", int], Awaitable[list[dict[str, Any]]]]
ALLOWED_FILTERS = {"service", "document_type", "environment"}
DEFAULT_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class RetrievalEvalCase:
    """One human-labeled retrieval question in the golden evaluation set."""

    case_id: str
    query: str
    relevant_chunks: frozenset[ChunkKey]
    filters: dict[str, str]
    expected_answer: str | None = None


@dataclass(frozen=True)
class StageSummary:
    """Aggregate retrieval metrics for one retrieval configuration."""

    name: str
    recall_at_k: dict[int, float]
    mrr: float
    first_relevant_ranks: dict[str, int | None]


def load_evaluation_cases(path: Path) -> list[RetrievalEvalCase]:
    """Load and validate retrieval evaluation cases from JSON."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Evaluation dataset must be a non-empty JSON array")

    cases: list[RetrievalEvalCase] = []
    seen_ids: set[str] = set()

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each evaluation case must be a JSON object")

        case_id = item.get("id")
        query = item.get("query")
        relevant = item.get("relevant_chunks")
        filters = item.get("filters", {})
        expected_answer = item.get("expected_answer")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each evaluation case needs a non-empty string id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {case_id!r} needs a non-empty query")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Evaluation case {case_id!r} needs at least one relevant chunk"
            )
        if not isinstance(filters, dict):
            raise ValueError(f"Evaluation case {case_id!r} filters must be an object")

        unsupported_filters = set(filters) - ALLOWED_FILTERS
        if unsupported_filters:
            names = ", ".join(sorted(unsupported_filters))
            raise ValueError(
                f"Evaluation case {case_id!r} has unsupported filters: {names}"
            )

        relevant_chunks: set[ChunkKey] = set()
        for chunk in relevant:
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Evaluation case {case_id!r} relevant chunks must be objects"
                )
            source = chunk.get("source")
            chunk_index = chunk.get("chunk_index")
            if not isinstance(source, str) or not isinstance(chunk_index, int):
                raise ValueError(
                    f"Evaluation case {case_id!r} relevant chunks need source and chunk_index"
                )
            relevant_chunks.add((source, chunk_index))

        normalized_filters: dict[str, str] = {}
        for name, value in filters.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Evaluation case {case_id!r} filter {name!r} must be a non-empty string"
                )
            normalized_filters[name] = value.strip().lower()

        if expected_answer is not None and not isinstance(expected_answer, str):
            raise ValueError(
                f"Evaluation case {case_id!r} expected_answer must be a string"
            )

        cases.append(
            RetrievalEvalCase(
                case_id=case_id,
                query=query,
                relevant_chunks=frozenset(relevant_chunks),
                filters=normalized_filters,
                expected_answer=expected_answer,
            )
        )

    return cases


def _result_key(result: Mapping[str, Any]) -> ChunkKey:
    return str(result["source"]), int(result["chunk_index"])


def recall_at_k(
    results: Sequence[Mapping[str, Any]],
    relevant_chunks: frozenset[ChunkKey],
    k: int,
) -> float:
    """Fraction of known-relevant chunks found in the first k results."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_chunks:
        raise ValueError("relevant_chunks must not be empty")

    retrieved = {_result_key(result) for result in results[:k]}
    return len(retrieved & relevant_chunks) / len(relevant_chunks)


def reciprocal_rank(
    results: Sequence[Mapping[str, Any]],
    relevant_chunks: frozenset[ChunkKey],
) -> float:
    """Return 1/rank for the first relevant result, or zero if none is retrieved."""

    if not relevant_chunks:
        raise ValueError("relevant_chunks must not be empty")

    for rank, result in enumerate(results, start=1):
        if _result_key(result) in relevant_chunks:
            return 1.0 / rank
    return 0.0


def first_relevant_rank(
    results: Sequence[Mapping[str, Any]],
    relevant_chunks: frozenset[ChunkKey],
) -> int | None:
    """Return the one-based rank of the first relevant result."""

    for rank, result in enumerate(results, start=1):
        if _result_key(result) in relevant_chunks:
            return rank
    return None


async def _vector_only(case: RetrievalEvalCase, limit: int) -> list[dict[str, Any]]:
    return await semantic_search_documents(query=case.query, limit=limit)


async def _vector_with_metadata(
    case: RetrievalEvalCase,
    limit: int,
) -> list[dict[str, Any]]:
    return await semantic_search_documents(
        query=case.query,
        limit=limit,
        **case.filters,
    )


async def _final_pipeline(
    case: RetrievalEvalCase,
    limit: int,
) -> list[dict[str, Any]]:
    return await hybrid_search_documents(
        query=case.query,
        limit=limit,
        **case.filters,
    )


async def evaluate_stage(
    name: str,
    retriever: Retriever,
    cases: Sequence[RetrievalEvalCase],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> StageSummary:
    """Evaluate one retrieval configuration over all golden questions."""

    if not cases:
        raise ValueError("cases must not be empty")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    max_k = max(k_values)
    if max_k > 10:
        raise ValueError("This project caps retrieval at 10 results; k cannot exceed 10")

    recall_totals = {k: 0.0 for k in k_values}
    reciprocal_rank_total = 0.0
    ranks: dict[str, int | None] = {}

    for case in cases:
        results = await retriever(case, max_k)

        for k in k_values:
            recall_totals[k] += recall_at_k(
                results,
                case.relevant_chunks,
                k,
            )

        reciprocal_rank_total += reciprocal_rank(results, case.relevant_chunks)
        ranks[case.case_id] = first_relevant_rank(results, case.relevant_chunks)

    count = len(cases)
    return StageSummary(
        name=name,
        recall_at_k={k: total / count for k, total in recall_totals.items()},
        mrr=reciprocal_rank_total / count,
        first_relevant_ranks=ranks,
    )


async def run_evaluation(
    cases: Sequence[RetrievalEvalCase],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> list[StageSummary]:
    """Compare the major retrieval stages built so far."""

    stages: list[tuple[str, Retriever]] = [
        ("vector-only", _vector_only),
        ("vector+metadata", _vector_with_metadata),
        ("hybrid+metadata+rerank", _final_pipeline),
    ]

    summaries: list[StageSummary] = []
    for name, retriever in stages:
        summaries.append(
            await evaluate_stage(
                name=name,
                retriever=retriever,
                cases=cases,
                k_values=k_values,
            )
        )
    return summaries


def print_summary(
    summaries: Sequence[StageSummary],
    k_values: Sequence[int],
) -> None:
    metric_headers = [f"Recall@{k}" for k in k_values]
    headers = ["Stage", *metric_headers, "MRR"]

    rows: list[list[str]] = []
    for summary in summaries:
        rows.append(
            [
                summary.name,
                *(f"{summary.recall_at_k[k]:.3f}" for k in k_values),
                f"{summary.mrr:.3f}",
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_case_details(
    cases: Sequence[RetrievalEvalCase],
    summaries: Sequence[StageSummary],
) -> None:
    print("\nFirst relevant rank by question")
    for case in cases:
        ranks = []
        for summary in summaries:
            rank = summary.first_relevant_ranks[case.case_id]
            ranks.append(f"{summary.name}={rank if rank is not None else 'miss'}")
        print(f"- {case.case_id}: " + ", ".join(ranks))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval against a human-labeled golden dataset."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="evals/retrieval.json",
        help="Path to the retrieval evaluation JSON file",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Recall cutoffs to report (default: 1 3 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the first relevant rank for every question",
    )
    args = parser.parse_args()

    k_values = tuple(dict.fromkeys(args.k))
    cases = load_evaluation_cases(Path(args.dataset))
    summaries = asyncio.run(run_evaluation(cases, k_values=k_values))

    print(f"Evaluated {len(cases)} golden questions from {args.dataset}\n")
    print_summary(summaries, k_values)
    if args.verbose:
        print_case_details(cases, summaries)


if __name__ == "__main__":
    main()
