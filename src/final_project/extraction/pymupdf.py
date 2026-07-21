"""Cleaned PyMuPDF fallback extraction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unicodedata

from unidecode import unidecode

from .errors import PdfExtractionError
from .models import ExtractedDocument, Section


SECTION_TYPES = {
    "SIMPLE SUMMARY": "abstract",
    "ABSTRACT": "abstract",
    "INTRODUCTION": "body",
    "BACKGROUND": "body",
    "METHODS": "body",
    "MATERIALS AND METHODS": "body",
    "MATERIAL AND METHODS": "body",
    "RESULTS": "body",
    "DISCUSSION": "body",
    "GENERAL DISCUSSION": "body",
    "CONCLUSION": "conclusion",
    "CONCLUSIONS": "conclusion",
    "REFERENCES": "references",
    "BIBLIOGRAPHY": "references",
}
SECTION_HEADING_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(item) for item in SECTION_TYPES) + r")\s*(?::\s*(.*))?$",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{2,90}$")


def clean_extracted_text(text: str) -> str:
    """Apply the validated PDF-specific corrective cleaning rules."""
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+\s+/\s+\d+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"doi:\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"^\s*\d+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_repeated_page_lines(
    page_texts: list[str], threshold: float = 0.7
) -> tuple[list[str], list[str]]:
    """Remove lines appearing on the configured fraction of distinct pages."""
    page_frequency: Counter[str] = Counter()
    for page_text in page_texts:
        page_frequency.update(
            {line.strip() for line in page_text.splitlines() if line.strip()}
        )
    minimum_pages = max(2, int(len(page_texts) * threshold + 0.999999))
    repeated = {
        line for line, count in page_frequency.items() if count >= minimum_pages
    }
    cleaned_pages = [
        "\n".join(
            line for line in page_text.splitlines() if line.strip() not in repeated
        )
        for page_text in page_texts
    ]
    return cleaned_pages, sorted(repeated)


def _looks_like_generic_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    words = stripped.split()
    return bool(NUMBERED_HEADING_RE.match(stripped)) or (
        stripped.isupper() and 1 <= len(words) <= 12
    )


def sections_from_text(text: str) -> tuple[str, list[Section], list[str]]:
    """Separate headings from prose and return ordered sections."""
    front_matter: list[str] = []
    sections: list[Section] = []
    references: list[str] = []
    current_kind = "front"
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        raw_lines = current_lines
        current_lines = []
        if current_kind == "references":
            references.extend(
                cleaned
                for cleaned in (clean_extracted_text(line) for line in raw_lines)
                if cleaned
            )
            return
        cleaned = clean_extracted_text("\n".join(raw_lines))
        if not cleaned:
            return
        if current_kind == "front":
            front_matter.append(cleaned)
        else:
            sections.append(
                Section(
                    section_type=current_kind,
                    heading=current_heading,
                    text=cleaned,
                    level=1,
                    order=len(sections),
                )
            )

    for line in text.splitlines():
        match = SECTION_HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(1).upper()
            current_kind = SECTION_TYPES[heading]
            current_heading = match.group(1).strip()
            extra_text = (match.group(2) or "").strip()
            if extra_text:
                current_lines.append(extra_text)
            continue
        if current_kind not in {"front", "references"} and _looks_like_generic_heading(line):
            flush()
            current_kind = "body"
            current_heading = line.strip()
            continue
        current_lines.append(line)
    flush()

    title = ""
    if front_matter:
        first_lines = [line.strip() for line in front_matter[0].splitlines() if line.strip()]
        title = first_lines[0] if first_lines else front_matter[0][:300]
    if title:
        sections.insert(0, Section("title", title, "", 0, 0))
        for index, section in enumerate(sections):
            section.order = index
    return title, sections, references


class PyMuPDFExtractor:
    """Extract and clean a PDF without an external service."""

    def __init__(self, repeated_line_threshold: float = 0.7) -> None:
        self.repeated_line_threshold = repeated_line_threshold

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        try:
            import pymupdf
        except ImportError as exc:
            raise PdfExtractionError("PyMuPDF is not installed") from exc

        pdf_path = Path(pdf_path).expanduser().resolve()
        if not pdf_path.is_file():
            raise PdfExtractionError(f"PDF does not exist: {pdf_path}")
        try:
            with pymupdf.open(pdf_path) as document:
                page_texts = [page.get_text() or "" for page in document]
                pdf_metadata = dict(document.metadata or {})
        except Exception as exc:
            raise PdfExtractionError(f"PyMuPDF failed for {pdf_path.name}: {exc}") from exc

        cleaned_pages, removed_lines = remove_repeated_page_lines(
            page_texts, self.repeated_line_threshold
        )
        title, sections, references = sections_from_text("\n".join(cleaned_pages))
        metadata_title = clean_extracted_text(pdf_metadata.get("title", ""))
        if metadata_title:
            title = metadata_title
            title_section = next(
                (section for section in sections if section.section_type == "title"),
                None,
            )
            if title_section:
                title_section.text = metadata_title
            else:
                sections.insert(0, Section("title", metadata_title, "", 0, 0))
                for index, section in enumerate(sections):
                    section.order = index
        return ExtractedDocument(
            document_id=pdf_path.stem,
            source_type="pdf",
            source_identifier=str(pdf_path),
            extraction_method="pymupdf_cleaned",
            metadata={
                "title": title,
                "authors": [],
                "doi": "",
                "journal": "",
                "publication_date": "",
            },
            sections=sections,
            references=references,
            diagnostics={
                "page_count": len(page_texts),
                "raw_character_count": sum(len(text) for text in page_texts),
                "character_count": sum(len(section.text) for section in sections),
                "removed_repeated_lines": removed_lines,
                "section_count": len(sections),
            },
            provenance={
                "requested_method": "pymupdf_cleaned",
                "fallback_used": False,
                "cleaning": "legacy_pdf_specific_cleanup",
            },
        )
