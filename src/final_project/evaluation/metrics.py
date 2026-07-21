"""Custom retrieval metrics for the seven-question benchmark."""

from __future__ import annotations

import re
from typing import Any


def normalize_source(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"\.[a-z0-9]+$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def source_matches(retrieved: str, expected: str) -> bool:
    retrieved_norm = normalize_source(retrieved)
    expected_norm = normalize_source(expected)
    if not retrieved_norm or not expected_norm:
        return False
    return retrieved_norm == expected_norm or expected_norm in retrieved_norm or retrieved_norm in expected_norm


def retrieved_source_name(chunk: dict[str, Any]) -> str:
    return (
        chunk.get("document_id")
        or chunk.get("title")
        or chunk.get("source_identifier")
        or ""
    )


def retrieved_source_aliases(chunk: dict[str, Any]) -> list[str]:
    aliases = [
        chunk.get("document_id", ""),
        chunk.get("title", ""),
        chunk.get("source_identifier", ""),
        chunk.get("doi", ""),
    ]
    return [str(alias) for alias in aliases if str(alias).strip()]


def unique_retrieved_sources(chunks: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    normalized_seen: set[str] = set()
    for chunk in chunks:
        source = retrieved_source_name(chunk)
        normalized = normalize_source(source)
        if normalized and normalized not in normalized_seen:
            normalized_seen.add(normalized)
            sources.append(source)
    return sources


def retrieval_metrics(
    chunks: list[dict[str, Any]],
    expected_sources: list[str],
) -> dict[str, float | int]:
    retrieved_sources = unique_retrieved_sources(chunks)
    expected = [source for source in expected_sources if source.strip()]
    if not expected:
        return {
            "retrieval_recall": 0.0,
            "retrieval_precision": 0.0,
            "mrr": 0.0,
            "expected_source_count": 0,
            "retrieved_source_count": len(retrieved_sources),
            "matched_source_count": 0,
        }

    matched_expected = set()
    first_match_rank = 0
    for rank, chunk in enumerate(chunks, start=1):
        aliases = retrieved_source_aliases(chunk)
        for expected_index, expected_source in enumerate(expected):
            if any(source_matches(alias, expected_source) for alias in aliases):
                matched_expected.add(expected_index)
                if first_match_rank == 0:
                    first_match_rank = rank

    precision = len(matched_expected) / len(retrieved_sources) if retrieved_sources else 0.0
    recall = len(matched_expected) / len(expected)
    mrr = 1.0 / first_match_rank if first_match_rank else 0.0
    return {
        "retrieval_recall": recall,
        "retrieval_precision": precision,
        "mrr": mrr,
        "expected_source_count": len(expected),
        "retrieved_source_count": len(retrieved_sources),
        "matched_source_count": len(matched_expected),
    }
