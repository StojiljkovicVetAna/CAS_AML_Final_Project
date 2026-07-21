"""Load the seven-question benchmark shared for final RAG evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvaluationQuestion:
    question_id: str
    question: str
    question_type: str = ""
    reference_answers: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)

    @property
    def reference_answer(self) -> str:
        return "\n\n".join(self.reference_answers)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference_answer"] = self.reference_answer
        return payload


def _clean_question_id(raw: str, fallback: int) -> str:
    value = (raw or "").strip()
    if not value:
        return str(fallback)
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def load_evaluation_questions(path: str | Path) -> list[EvaluationQuestion]:
    """Group the NotebookLM-style CSV rows into one item per numbered question."""
    csv_path = Path(path).expanduser().resolve()
    questions: list[EvaluationQuestion] = []
    current: EvaluationQuestion | None = None

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question_nr = (row.get("Question Nr") or "").strip()
            question_text = (row.get("Question") or "").strip()
            if question_nr or question_text:
                current = EvaluationQuestion(
                    question_id=_clean_question_id(question_nr, len(questions) + 1),
                    question=question_text,
                    question_type=(row.get("Type (narrow/broad)") or "").strip(),
                )
                questions.append(current)

            if current is None:
                continue

            answer = (row.get("Answer") or "").strip()
            source = (row.get("Source") or "").strip()
            if answer:
                current.reference_answers.append(answer)
            if source and source not in current.expected_sources:
                current.expected_sources.append(source)

    return questions


def write_dataset_json(
    questions: list[EvaluationQuestion],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([question.to_dict() for question in questions], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path
