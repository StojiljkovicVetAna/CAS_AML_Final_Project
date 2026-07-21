"""Chunking schema for records that will be embedded."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


CHUNK_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class ChunkConfig:
    """Default chunking configuration for academic RAG retrieval."""

    target_tokens: int = 400
    max_tokens: int = 520
    overlap_tokens: int = 80
    min_tokens: int = 80
    include_section_types: tuple[str, ...] = ("abstract", "body", "conclusion")
    exclude_headings: tuple[str, ...] = (
        "references",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "supplementary material",
        "supporting information",
    )


@dataclass(slots=True)
class TextChunk:
    """One text unit prepared for embedding and vector storage."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    section_type: str
    heading: str
    section_order: int
    extraction_method: str
    source_type: str
    source_identifier: str
    token_count: int
    sentence_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provenance.setdefault(
            "chunked_at", datetime.now(timezone.utc).isoformat()
        )
        self.provenance.setdefault("chunk_schema_version", CHUNK_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
