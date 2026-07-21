"""Collect evaluation summaries into one report-ready table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final RAG evaluation summaries")
    parser.add_argument("--evaluation-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument(
        "--extra-evaluation-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional evaluation directory to merge, such as a single-strategy follow-up run.",
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("data/evaluation/final_evaluation_summary"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: str) -> float | str:
    if value in {"", "None", "nan", "NaN"}:
        return ""
    try:
        return round(float(value), 4)
    except ValueError:
        return value


def _prefix_row(row: dict[str, str], prefix: str, skip: set[str]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": _as_float(value)
        for key, value in row.items()
        if key not in skip
    }


def _collect_into(evaluation_dir: Path, by_strategy: dict[str, dict[str, Any]]) -> None:
    classical_rows = read_csv(evaluation_dir / "rag_strategy_summary.csv")
    judge_rows = read_csv(evaluation_dir / "llm_judge" / "llm_judge_summary.csv")
    ragas_rows = read_csv(evaluation_dir / "ragas" / "ragas_summary.csv")

    for row in classical_rows:
        strategy = row["strategy"]
        by_strategy.setdefault(strategy, {"strategy": strategy})
        by_strategy[strategy].update(_prefix_row(row, "custom", {"strategy"}))
    for row in judge_rows:
        strategy = row["strategy"]
        by_strategy.setdefault(strategy, {"strategy": strategy})
        by_strategy[strategy].update(_prefix_row(row, "judge", {"strategy"}))
    for row in ragas_rows:
        strategy = row["strategy"]
        by_strategy.setdefault(strategy, {"strategy": strategy})
        by_strategy[strategy].update(_prefix_row(row, "ragas", {"strategy"}))


def collect(evaluation_dirs: Path | list[Path]) -> list[dict[str, Any]]:
    if isinstance(evaluation_dirs, Path):
        evaluation_dirs = [evaluation_dirs]
    by_strategy: dict[str, dict[str, Any]] = {}
    for evaluation_dir in evaluation_dirs:
        _collect_into(evaluation_dir, by_strategy)
    return [by_strategy[strategy] for strategy in sorted(by_strategy)]


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No evaluation rows found.\n", encoding="utf-8")
        return

    preferred_columns = [
        "strategy",
        "custom_mean_retrieval_recall",
        "custom_mean_retrieval_precision",
        "custom_mean_mrr",
        "custom_mean_matched_source_count",
        "judge_mean_answer_correctness",
        "judge_mean_faithfulness",
        "judge_mean_context_usefulness",
        "judge_mean_overall_quality",
        "ragas_mean_faithfulness",
        "ragas_mean_llm_context_precision_with_reference",
        "ragas_mean_context_recall",
    ]
    columns = [column for column in preferred_columns if any(column in row for row in rows)]
    lines = [
        "# Final Evaluation Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- `custom_*` metrics are deterministic paper-level retrieval metrics from the benchmark source labels.",
            "- `judge_*` metrics are 1-5 LLM-as-judge scores.",
            "- Empty `ragas_*` values mean RAGAS did not return valid scores for that metric.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    evaluation_dirs = [args.evaluation_dir.expanduser()] + [
        path.expanduser() for path in args.extra_evaluation_dir
    ]
    output_prefix = args.output_prefix.expanduser()
    rows = collect(evaluation_dirs)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_markdown(output_prefix.with_suffix(".md"), rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} strategy summaries to {output_prefix.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
