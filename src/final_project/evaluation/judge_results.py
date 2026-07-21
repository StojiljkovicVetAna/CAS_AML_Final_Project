"""LLM-as-judge evaluation for saved RAG experiment outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import logging
from pathlib import Path
import re
from statistics import mean
import time
from typing import Any

from dotenv import load_dotenv

from final_project.backend.config import active_llm_model, load_settings
from final_project.backend.llm import generate_text


LOGGER = logging.getLogger(__name__)

SCORE_FIELDS = [
    "answer_correctness",
    "faithfulness",
    "answer_relevance",
    "context_usefulness",
    "completeness",
    "overall_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score RAG answers with an LLM judge")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evaluation/rag_strategy_results.jsonl"),
        help="JSONL produced by rag-evaluate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/llm_judge"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--judge-max-tokens", type=int, default=4000)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-wait-seconds", type=float, default=90.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing judge outputs and score every input row again",
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


def context_text(chunks: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("document_id") or chunk.get("title") or "unknown source"
        heading = chunk.get("heading") or chunk.get("section_type") or ""
        blocks.append(f"[Source {index}: {source}; {heading}]\n{chunk.get('text', '')}")
    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n[Context truncated for judging]"


def build_judge_prompt(record: dict[str, Any], max_context_chars: int) -> str:
    benchmark = record.get("benchmark", {})
    expected_sources = benchmark.get("expected_sources", [])
    reference_answer = benchmark.get("reference_answer", "")
    context = context_text(record.get("chunks", []), max_context_chars)
    return f"""You are evaluating a scientific RAG answer about dog behaviour.
Judge ONLY the answer quality using the provided question, reference answer, expected sources, and retrieved context.

Return ONLY valid JSON with this schema:
{{
  "answer_correctness": 1,
  "faithfulness": 1,
  "answer_relevance": 1,
  "context_usefulness": 1,
  "completeness": 1,
  "overall_quality": 1,
  "reason": "brief explanation"
}}

Scoring scale for every numeric field:
1 = poor, 2 = weak, 3 = acceptable, 4 = good, 5 = excellent.

Rubric:
- answer_correctness: factual agreement with the reference answer.
- faithfulness: whether the answer is supported by the retrieved context and does not add unsupported claims.
- answer_relevance: whether the answer directly addresses the question.
- context_usefulness: whether the retrieved context contains useful evidence for the answer.
- completeness: whether the answer covers the important parts of the reference answer.
- overall_quality: holistic score considering correctness, grounding, relevance, and completeness.

Question:
{record.get("question", "")}

Strategy:
{record.get("strategy", "")}

Expected sources:
{json.dumps(expected_sources, ensure_ascii=False)}

Reference answer:
{reference_answer}

Retrieved context:
{context}

Generated answer:
{record.get("answer", "")}
"""


def parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])

    parsed: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        value = payload.get(field, 0)
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        parsed[field] = min(5.0, max(0.0, score))
    parsed["reason"] = str(payload.get("reason", "")).strip()
    return parsed


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


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("question_id", "")), str(record.get("strategy", ""))


def _is_rate_limit_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def _retry_after_seconds(exc: Exception, default: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(default, float(retry_after))
        except ValueError:
            return default
    return default


def load_existing_outputs(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = output_dir / "llm_judge_results.jsonl"
    if not path.exists():
        return [], []

    judged_records = load_records(path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    deduped_records: list[dict[str, Any]] = []
    for record in judged_records:
        key = _record_key(record)
        if key in seen:
            continue
        parsed = record.get("llm_judge")
        if not isinstance(parsed, dict):
            continue
        seen.add(key)
        deduped_records.append(record)
        rows.append(
            {
                "strategy": record.get("strategy", ""),
                "question_id": record.get("question_id", ""),
                "question": record.get("question", ""),
                **{field: parsed.get(field, 0.0) for field in SCORE_FIELDS},
                "reason": parsed.get("reason", ""),
                "judge_raw": record.get("llm_judge_raw", ""),
            }
        )
    return deduped_records, rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)

    summary_rows = []
    for strategy, strategy_rows in sorted(by_strategy.items()):
        summary = {"strategy": strategy, "n_questions": len(strategy_rows)}
        for field in SCORE_FIELDS:
            summary[f"mean_{field}"] = mean(float(row[field]) for row in strategy_rows)
        summary_rows.append(summary)
    return summary_rows


def judge_record(
    record: dict[str, Any],
    *,
    settings: Any,
    max_context_chars: int,
    max_retries: int,
    retry_wait_seconds: float,
) -> tuple[dict[str, Any], str]:
    """Judge one record, shrinking context on prompt issues and waiting on rate limits."""
    context_budgets = [max_context_chars, max(3000, max_context_chars // 2)]
    last_error: Exception | None = None
    for budget in context_budgets:
        for attempt in range(1, max_retries + 1):
            try:
                prompt = build_judge_prompt(record, budget)
                raw = generate_text(prompt, settings=settings)
                return parse_judge_json(raw), raw
            except Exception as exc:  # noqa: BLE001 - preserve the original failure in the final error.
                last_error = exc
                if _is_rate_limit_error(exc) and attempt < max_retries:
                    wait_seconds = _retry_after_seconds(exc, retry_wait_seconds)
                    LOGGER.warning(
                        "Judge rate limit for q%s %s with context budget %s; waiting %.1f seconds before retry %s/%s.",
                        record.get("question_id"),
                        record.get("strategy"),
                        budget,
                        wait_seconds,
                        attempt,
                        max_retries,
                    )
                    time.sleep(wait_seconds)
                    continue
                LOGGER.warning(
                    "Judge attempt failed for q%s %s with context budget %s: %s",
                    record.get("question_id"),
                    record.get("strategy"),
                    budget,
                    exc,
                )
                break
    raise RuntimeError(
        f"Could not judge q{record.get('question_id')} {record.get('strategy')}: {last_error}"
    ) from last_error


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    settings = load_settings()
    if args.judge_max_tokens:
        settings = replace(
            settings,
            llm_max_tokens=max(settings.llm_max_tokens, args.judge_max_tokens),
        )
    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_resume:
        judged_records: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
    else:
        judged_records, rows = load_existing_outputs(output_dir)
        if judged_records:
            LOGGER.info("Resuming from %s existing judged rows.", len(judged_records))
    judged_keys = {_record_key(record) for record in judged_records}
    for index, record in enumerate(records, start=1):
        if _record_key(record) in judged_keys:
            LOGGER.info(
                "Skipping already judged %s/%s: q%s %s",
                index,
                len(records),
                record.get("question_id"),
                record.get("strategy"),
            )
            continue
        LOGGER.info(
            "Judging %s/%s: q%s %s",
            index,
            len(records),
            record.get("question_id"),
            record.get("strategy"),
        )
        parsed, raw = judge_record(
            record,
            settings=settings,
            max_context_chars=args.max_context_chars,
            max_retries=args.max_retries,
            retry_wait_seconds=args.retry_wait_seconds,
        )
        judged = {
            "strategy": record.get("strategy", ""),
            "question_id": record.get("question_id", ""),
            "question": record.get("question", ""),
            **parsed,
            "judge_raw": raw,
        }
        rows.append(judged)
        judged_records.append({**record, "llm_judge": parsed, "llm_judge_raw": raw})
        judged_keys.add(_record_key(record))
        _write_jsonl(output_dir / "llm_judge_results.jsonl", judged_records)
        _write_csv(output_dir / "llm_judge_scores.csv", rows)

    summary_rows = summarize(rows)
    _write_jsonl(output_dir / "llm_judge_results.jsonl", judged_records)
    _write_csv(output_dir / "llm_judge_scores.csv", rows)
    _write_csv(output_dir / "llm_judge_summary.csv", summary_rows)
    (output_dir / "judge_manifest.json").write_text(
        json.dumps(
            {
                "input": str(args.input.expanduser().resolve()),
                "output_dir": str(output_dir),
                "n_records": len(records),
                "llm_provider": settings.llm_provider,
                "llm_model": active_llm_model(settings),
                "score_fields": SCORE_FIELDS,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote LLM judge outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
