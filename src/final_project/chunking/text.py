"""Text normalization, sentence splitting, and filtering for chunking."""

from __future__ import annotations

import math
import re


PERIOD_PLACEHOLDER = "<prd>"

ACADEMIC_ABBREVIATIONS = (
    "et al.",
    "e.g.",
    "i.e.",
    "cf.",
    "vs.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Table.",
    "Tab.",
    "Dr.",
    "Prof.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "St.",
    "No.",
    "Vol.",
    "Inc.",
    "Ltd.",
)

CAPTION_PATTERNS = (
    re.compile(r"^\s*(fig(?:ure)?|table)\.?\s*\d+[\s:.-]", re.IGNORECASE),
)

SUPPLEMENTARY_NOTICE_PATTERNS = (
    re.compile(r"^\s*supplementary\s+(material|file|figure|table|information)\b", re.IGNORECASE),
    re.compile(r"^\s*additional\s+file\b", re.IGNORECASE),
    re.compile(r"^\s*electronic\s+supplementary\s+material\b", re.IGNORECASE),
    re.compile(r"^\s*the\s+online\s+version\s+contains\s+supplementary\s+material\b", re.IGNORECASE),
    re.compile(r"^\s*below\s+is\s+the\s+link\s+to\s+the\s+electronic\s+supplementary\s+material\b", re.IGNORECASE),
)

PROSE_FIGURE_VERBS = re.compile(
    r"\b(shows?|illustrates?|depicts?|presents?|reports?|summari[sz]es?|displays?|indicates?)\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse local whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_token_count(text: str) -> int:
    """Approximate embedding-token length without adding tokenizer dependencies."""
    if not text.strip():
        return 0
    word_like = re.findall(r"\b\w+(?:[-']\w+)?\b|[^\w\s]", text, flags=re.UNICODE)
    return max(len(word_like), math.ceil(len(text) / 4))


def split_paragraphs(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]


def is_caption_or_notice(text: str) -> bool:
    """Return True for paragraphs/sentences that should not be embedded."""
    stripped = normalize_whitespace(text)
    if not stripped:
        return True
    if any(pattern.search(stripped) for pattern in CAPTION_PATTERNS):
        return True
    if is_supplementary_notice(stripped):
        return True
    return False


def is_standalone_caption_sentence(text: str) -> bool:
    """Detect short label-style figure/table captions embedded in prose."""
    stripped = normalize_whitespace(text)
    if not any(pattern.search(stripped) for pattern in CAPTION_PATTERNS):
        return False
    return PROSE_FIGURE_VERBS.search(stripped) is None


def is_supplementary_notice(text: str) -> bool:
    stripped = normalize_whitespace(text)
    if not stripped:
        return True
    if any(pattern.search(stripped) for pattern in SUPPLEMENTARY_NOTICE_PATTERNS):
        return True
    lowered = stripped.lower()
    return (
        lowered.startswith("supplementary file")
        or lowered.startswith("supplementary data")
        or lowered.startswith("supporting information")
    )


def _protect_abbreviations(text: str) -> str:
    protected = text
    for abbreviation in ACADEMIC_ABBREVIATIONS:
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group(0).replace(".", PERIOD_PLACEHOLDER),
            protected,
            flags=re.IGNORECASE,
        )
    protected = re.sub(
        r"\b([A-Z])\.(?=\s+[A-Z]\.)",
        lambda match: f"{match.group(1)}{PERIOD_PLACEHOLDER}",
        protected,
    )
    return protected


def _restore_abbreviations(text: str) -> str:
    return text.replace(PERIOD_PLACEHOLDER, ".")


def split_sentences(text: str) -> list[str]:
    """Split academic prose while avoiding common abbreviation breaks."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    protected = _protect_abbreviations(normalized)
    raw_sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])",
        protected,
    )
    sentences = []
    for raw in raw_sentences:
        sentence = _restore_abbreviations(raw).strip()
        if (
            sentence
            and not is_supplementary_notice(sentence)
            and not is_standalone_caption_sentence(sentence)
        ):
            sentences.append(sentence)
    return sentences
