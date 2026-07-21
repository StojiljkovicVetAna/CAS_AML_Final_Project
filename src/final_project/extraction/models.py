"""Common output schema shared by every extraction method."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class Section:
    """One ordered logical section of an extracted paper."""

    section_type: str
    text: str
    heading: str = ""
    level: int = 1
    order: int = 0


@dataclass(slots=True)
class ExtractedDocument:
    """Normalized extraction result consumed by later RAG modules."""

    document_id: str
    source_type: str
    source_identifier: str
    extraction_method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_xml: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.provenance.setdefault(
            "extracted_at", datetime.now(timezone.utc).isoformat()
        )
        self.provenance.setdefault("schema_version", SCHEMA_VERSION)

    def full_text(self, include_references: bool = False) -> str:
        """Return ordered section text for downstream chunking."""
        parts = [section.text for section in self.sections if section.text.strip()]
        if include_references:
            parts.extend(reference for reference in self.references if reference.strip())
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize without embedding raw XML in the normalized JSON."""
        return {
            "schema_version": SCHEMA_VERSION,
            "document_id": self.document_id,
            "source_type": self.source_type,
            "source_identifier": self.source_identifier,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
            "sections": [asdict(section) for section in self.sections],
            "references": self.references,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "provenance": self.provenance,
        }
