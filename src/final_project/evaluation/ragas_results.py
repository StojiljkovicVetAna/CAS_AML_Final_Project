"""Run RAGAS metrics on saved RAG experiment outputs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from final_project.backend.config import active_llm_model, load_settings
from final_project.backend.llm import generate_text


LOGGER = logging.getLogger(__name__)

METRIC_FACTORIES = {
    "faithfulness": "Faithfulness",
    "context_precision": "LLMContextPrecisionWithReference",
    "context_recall": "LLMContextRecall",
    "answer_correctness": "AnswerCorrectness",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved RAG outputs with RAGAS")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evaluation/rag_strategy_results.jsonl"),
        help="JSONL produced by rag-evaluate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/ragas"))
    parser.add_argument(
        "--metrics",
        default="faithfulness,context_precision,context_recall",
        help=(
            "Comma-separated metrics. Supported: faithfulness, context_precision, "
            "context_recall, answer_correctness"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--max-reference-chars", type=int, default=4000)
    parser.add_argument("--llm-max-tokens", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--raise-exceptions", action="store_true")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit successfully even if RAGAS returns no valid metric values",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [truncated]"


def record_to_sample_payload(
    record: dict[str, Any],
    max_context_chars: int,
    max_reference_chars: int,
) -> dict[str, Any]:
    contexts = [
        truncate_context(str(chunk.get("text", "")), max_context_chars)
        for chunk in record.get("chunks", [])
        if str(chunk.get("text", "")).strip()
    ]
    benchmark = record.get("benchmark", {})
    return {
        "user_input": record.get("question", ""),
        "response": record.get("answer", ""),
        "retrieved_contexts": contexts,
        "reference": truncate_context(str(benchmark.get("reference_answer", "")), max_reference_chars),
    }


def make_metrics(metric_names: list[str]) -> list[Any]:
    from ragas import metrics as ragas_metrics

    metrics = []
    for name in metric_names:
        factory_name = METRIC_FACTORIES.get(name)
        if not factory_name:
            raise ValueError(f"Unsupported RAGAS metric: {name}")
        metrics.append(getattr(ragas_metrics, factory_name)())
    return metrics


def make_llm(settings: Any, max_tokens: int) -> Any:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for RAGAS evaluation")
        return ResponsesApiRagasLLM(settings=settings, max_tokens=max_tokens)

    if settings.llm_provider != "openai_compatible":
        raise ValueError("RAGAS currently expects LLM_PROVIDER=openai or openai_compatible")
    if not settings.openai_compatible_api_key:
        raise ValueError("OPENAI_COMPATIBLE_API_KEY is required for RAGAS evaluation")
    if not settings.openai_compatible_base_url:
        raise ValueError("OPENAI_COMPATIBLE_BASE_URL is required for RAGAS evaluation")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_compatible_model or settings.llm_model_name,
        api_key=settings.openai_compatible_api_key,
        base_url=settings.openai_compatible_base_url,
        temperature=0,
        max_tokens=max_tokens,
    )


class ResponsesApiRagasLLM:
    """Minimal RAGAS LLM wrapper that uses this project's Responses API client."""

    def __init__(self, *, settings: Any, max_tokens: int):
        from dataclasses import replace
        from ragas.run_config import RunConfig

        self.settings = replace(settings, llm_max_tokens=max(settings.llm_max_tokens, max_tokens))
        self.run_config = RunConfig()
        self.multiple_completion_supported = False
        self.cache = None

    def set_run_config(self, run_config: Any) -> None:
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        return 0.3 if n > 1 else 0.01

    @staticmethod
    def _prompt_to_text(prompt: Any) -> str:
        if hasattr(prompt, "to_string"):
            return str(prompt.to_string())
        if hasattr(prompt, "text"):
            return str(prompt.text)
        return str(prompt)

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> Any:
        from langchain_core.outputs import Generation, LLMResult

        prompt_text = self._prompt_to_text(prompt)
        generations = []
        for _ in range(max(1, n)):
            text = generate_text(prompt_text, settings=self.settings)
            generations.append(Generation(text=text, generation_info={"finish_reason": "stop"}))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.generate_text,
            prompt,
            n,
            temperature,
            stop,
            callbacks,
        )

    async def generate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> Any:
        if temperature is None:
            temperature = self.get_temperature(n)
        return await self.agenerate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    def is_finished(self, response: Any) -> bool:
        return True


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], metric_columns: list[str]) -> list[dict[str, Any]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)

    summary_rows = []
    for strategy, strategy_rows in sorted(by_strategy.items()):
        summary: dict[str, Any] = {"strategy": strategy, "n_questions": len(strategy_rows)}
        for column in metric_columns:
            values = [
                float(row[column])
                for row in strategy_rows
                if row.get(column) not in {"", None}
                and str(row.get(column)).lower() not in {"nan", "none"}
            ]
            summary[f"n_valid_{column}"] = len(values)
            summary[f"mean_{column}"] = mean(values) if values else ""
        summary_rows.append(summary)
    return summary_rows


def has_valid_metric(summary_rows: list[dict[str, Any]], metric_columns: list[str]) -> bool:
    for row in summary_rows:
        for column in metric_columns:
            if int(row.get(f"n_valid_{column}", 0) or 0) > 0:
                return True
    return False


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    metrics = make_metrics(metric_names)
    settings = load_settings()
    llm = make_llm(settings, args.llm_max_tokens)

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig

    sample_payloads = [
        record_to_sample_payload(record, args.max_context_chars, args.max_reference_chars)
        for record in records
    ]
    _write_jsonl(output_dir / "ragas_dataset.jsonl", sample_payloads)
    dataset = EvaluationDataset(samples=[SingleTurnSample(**payload) for payload in sample_payloads])

    run_config = RunConfig(max_workers=args.max_workers)
    if hasattr(llm, "set_run_config"):
        llm.set_run_config(run_config)
    LOGGER.info("Running RAGAS on %s records with metrics: %s", len(records), ", ".join(metric_names))
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        run_config=run_config,
        raise_exceptions=args.raise_exceptions,
        show_progress=True,
        batch_size=args.batch_size,
    )

    frame = result.to_pandas()
    metric_columns = [column for column in frame.columns if column not in sample_payloads[0]]
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        record = records[index]
        payload = {
            "strategy": record.get("strategy", ""),
            "question_id": record.get("question_id", ""),
            "question": record.get("question", ""),
        }
        for column in metric_columns:
            value = row[column]
            payload[column] = "" if str(value).lower() == "nan" else value
        rows.append(payload)

    summary_rows = summarize(rows, metric_columns)
    _write_csv(output_dir / "ragas_scores.csv", rows)
    _write_csv(output_dir / "ragas_summary.csv", summary_rows)
    (output_dir / "ragas_manifest.json").write_text(
        json.dumps(
            {
                "input": str(args.input.expanduser().resolve()),
                "output_dir": str(output_dir),
                "n_records": len(records),
                "metrics": metric_names,
                "llm_provider": settings.llm_provider,
                "llm_model": active_llm_model(settings),
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not has_valid_metric(summary_rows, metric_columns) and not args.allow_empty:
        raise RuntimeError(
            "RAGAS completed but returned zero valid metric values. "
            "The CSV files were written for inspection. Rerun with --raise-exceptions "
            "to expose the provider/model error, or use rag-collect-results to report "
            "the deterministic custom metrics and LLM-judge scores."
        )
    LOGGER.info("Wrote RAGAS outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
