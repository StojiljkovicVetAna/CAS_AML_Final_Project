"""Run the retrieval-strategy RAG evaluation experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import logging
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from final_project.backend.config import active_llm_model, load_settings
from final_project.evaluation.dataset import EvaluationQuestion, load_evaluation_questions
from final_project.evaluation.metrics import retrieval_metrics, unique_retrieved_sources
from final_project.evaluation.strategies import (
    StrategyResult,
    run_agentic_rag,
    run_classic_rag,
    run_hyde_reranked_rag,
    run_hyde_rag,
    run_multi_query_rag,
    run_reranked_rag,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG retrieval strategies")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../rag-framework/FINAL_Q and A_Step1.csv"),
        help="CSV with Question Nr, Question, Answer, and Source columns",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument(
        "--strategies",
        default="classic,reranked,hyde,hyde_reranked,multi_query",
        help="Comma-separated subset: classic,reranked,hyde,hyde_reranked,multi_query,agentic",
    )
    parser.add_argument("--context-k", type=int, default=6)
    parser.add_argument("--rerank-candidate-k", type=int, default=20)
    parser.add_argument("--hyde-sentence-count", type=int, default=10)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--multi-query-candidate-k", type=int, default=10)
    parser.add_argument("--agent-initial-k", type=int, default=6)
    parser.add_argument("--agent-second-candidate-k", type=int, default=20)
    parser.add_argument("--agent-rerank-second", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _result_row(
    question: EvaluationQuestion,
    result: StrategyResult,
) -> dict[str, object]:
    metrics = retrieval_metrics(result.chunks, question.expected_sources)
    return {
        "strategy": result.strategy,
        "question_id": question.question_id,
        "question": question.question,
        "question_type": question.question_type,
        "answer": result.answer,
        "reference_answer": question.reference_answer,
        "expected_sources": json.dumps(question.expected_sources, ensure_ascii=False),
        "retrieved_sources": json.dumps(unique_retrieved_sources(result.chunks), ensure_ascii=False),
        "retrieval_query": result.retrieval_query,
        "retrieval_queries": json.dumps(result.retrieval_queries or [result.retrieval_query], ensure_ascii=False),
        "rewritten_query": result.rewritten_query,
        "context_sufficient": result.context_sufficient,
        "second_retrieval_used": result.second_retrieval_used,
        "num_chunks": len(result.chunks),
        **metrics,
        "timings_ms": json.dumps(result.timings_ms, ensure_ascii=False),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_strategy: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)
    summary_rows = []
    for strategy, strategy_rows in sorted(by_strategy.items()):
        summary_rows.append(
            {
                "strategy": strategy,
                "n_questions": len(strategy_rows),
                "mean_retrieval_recall": mean(float(row["retrieval_recall"]) for row in strategy_rows),
                "mean_retrieval_precision": mean(float(row["retrieval_precision"]) for row in strategy_rows),
                "mean_mrr": mean(float(row["mrr"]) for row in strategy_rows),
                "mean_matched_source_count": mean(float(row["matched_source_count"]) for row in strategy_rows),
                "second_retrieval_rate": mean(
                    1.0 if row["second_retrieval_used"] in {True, "True", "true"} else 0.0
                    for row in strategy_rows
                ),
            }
        )
    return summary_rows


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    settings = load_settings()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_evaluation_questions(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
    strategies = {
        item.strip().lower().replace("-", "_")
        for item in args.strategies.split(",")
        if item.strip()
    }

    result_records: list[dict] = []
    rows: list[dict[str, object]] = []

    for question in questions:
        LOGGER.info("Evaluating question %s: %s", question.question_id, question.question)
        results: list[StrategyResult] = []
        if "classic" in strategies:
            results.append(
                run_classic_rag(
                    question_id=question.question_id,
                    question=question.question,
                    settings=settings,
                    context_k=args.context_k,
                )
            )
        if "reranked" in strategies:
            results.append(
                run_reranked_rag(
                    question_id=question.question_id,
                    question=question.question,
                    settings=settings,
                    candidate_k=args.rerank_candidate_k,
                    context_k=args.context_k,
                )
            )
        if "hyde" in strategies:
            results.append(
                run_hyde_rag(
                    question_id=question.question_id,
                    question=question.question,
                    settings=settings,
                    context_k=args.context_k,
                    hyde_sentence_count=args.hyde_sentence_count,
                )
            )
        if "hyde_reranked" in strategies:
            results.append(
                run_hyde_reranked_rag(
                    question_id=question.question_id,
                    question=question.question,
                    settings=settings,
                    candidate_k=args.rerank_candidate_k,
                    context_k=args.context_k,
                    hyde_sentence_count=args.hyde_sentence_count,
                )
            )
        if "multi_query" in strategies:
            results.append(
                run_multi_query_rag(
                    question_id=question.question_id,
                    question=question.question,
                    settings=settings,
                    context_k=args.context_k,
                    num_queries=args.multi_query_count,
                    per_query_k=args.multi_query_candidate_k,
                )
            )
        if "agentic" in strategies:
            results.append(
                run_agentic_rag(
                    question_id=question.question_id,
                    question=question.question,
                    settings=settings,
                    initial_k=args.agent_initial_k,
                    context_k=args.context_k,
                    second_candidate_k=args.agent_second_candidate_k,
                    rerank_second_retrieval=args.agent_rerank_second,
                )
            )

        for result in results:
            result_payload = result.to_dict()
            result_payload["benchmark"] = question.to_dict()
            result_records.append(result_payload)
            rows.append(_result_row(question, result))

    summary_rows = _summary(rows)
    _write_jsonl(output_dir / "rag_strategy_results.jsonl", result_records)
    _write_csv(output_dir / "rag_strategy_results.csv", rows)
    _write_csv(output_dir / "rag_strategy_summary.csv", summary_rows)
    (output_dir / "benchmark_questions.json").write_text(
        json.dumps([question.to_dict() for question in questions], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.expanduser().resolve()),
                "output_dir": str(output_dir),
                "strategies": sorted(strategies),
                "context_k": args.context_k,
                "rerank_candidate_k": args.rerank_candidate_k,
                "hyde_sentence_count": args.hyde_sentence_count,
                "multi_query_count": args.multi_query_count,
                "multi_query_candidate_k": args.multi_query_candidate_k,
                "agent_initial_k": args.agent_initial_k,
                "agent_second_candidate_k": args.agent_second_candidate_k,
                "agent_rerank_second": args.agent_rerank_second,
                "settings": {
                    "chroma_mode": settings.chroma_mode,
                    "chroma_collection": settings.chroma_collection,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "reranker_provider": settings.reranker_provider,
                    "llm_provider": settings.llm_provider,
                    "llm_model": active_llm_model(settings),
                },
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote evaluation outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
