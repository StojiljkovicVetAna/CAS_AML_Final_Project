"""GROBID service client and TEI parser."""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

from .errors import GrobidError
from .models import ExtractedDocument, Section
from .xml_utils import (
    attribute_case_insensitive,
    descendants,
    direct_children,
    element_text,
    first_descendant,
    paragraph_text,
)


def _first_text_with_attribute(
    parent: ET.Element | None,
    element_name: str,
    attribute: str,
    accepted_values: set[str],
) -> str:
    for element in descendants(parent, element_name):
        value = attribute_case_insensitive(element, attribute).lower()
        if value in accepted_values:
            text = element_text(element)
            if text:
                return text
    return ""


def _tei_authors(root: ET.Element) -> list[str]:
    analytic = first_descendant(root, "analytic")
    authors = []
    for author in descendants(analytic, "author"):
        pers_name = first_descendant(author, "persName") or author
        forenames = [element_text(item) for item in descendants(pers_name, "forename")]
        surname = element_text(first_descendant(pers_name, "surname"))
        name = " ".join(part for part in [*forenames, surname] if part).strip()
        if name and name not in authors:
            authors.append(name)
    return authors


def _section_type(heading: str) -> str:
    normalized = heading.lower()
    if "conclusion" in normalized:
        return "conclusion"
    return "body"


def _tei_body_sections(root: ET.Element) -> list[Section]:
    body = first_descendant(first_descendant(root, "text"), "body")
    if body is None:
        return []

    sections: list[Section] = []

    def add_section(kind: str, heading: str, text: str, level: int) -> None:
        if not text.strip():
            return
        sections.append(
            Section(
                section_type=kind,
                heading=heading,
                text=text,
                level=level,
                order=len(sections),
            )
        )

    def walk_div(div: ET.Element, level: int) -> None:
        heading = element_text(next(iter(direct_children(div, "head")), None))
        text = paragraph_text(direct_children(div, "p"))
        add_section(_section_type(heading), heading, text, level)
        for child_div in direct_children(div, "div"):
            walk_div(child_div, level + 1)

    direct_paragraphs = paragraph_text(direct_children(body, "p"))
    add_section("body", "", direct_paragraphs, 1)
    for div in direct_children(body, "div"):
        walk_div(div, 1)

    if not sections:
        add_section("body", "", paragraph_text(descendants(body, "p")), 1)
    return sections


def parse_tei(tei_xml: str, source_identifier: str) -> ExtractedDocument:
    """Parse GROBID TEI into the normalized extraction schema."""
    try:
        root = ET.fromstring(tei_xml.encode("utf-8"))
    except ET.ParseError as exc:
        raise GrobidError(f"GROBID returned invalid TEI XML: {exc}") from exc

    analytic = first_descendant(root, "analytic")
    title = _first_text_with_attribute(
        analytic, "title", "level", {"a"}
    ) or _first_text_with_attribute(root, "title", "type", {"main"})
    if not title:
        title = element_text(first_descendant(root, "title"))

    abstract = paragraph_text(descendants(first_descendant(root, "abstract"), "p"))
    body_sections = _tei_body_sections(root)
    sections: list[Section] = []
    if title:
        sections.append(Section("title", title, "", 0, len(sections)))
    if abstract:
        sections.append(Section("abstract", abstract, "Abstract", 1, len(sections)))
    for section in body_sections:
        section.order = len(sections)
        sections.append(section)

    references = []
    back = first_descendant(first_descendant(root, "text"), "back")
    for bibliography in descendants(back, "biblStruct"):
        text = element_text(bibliography)
        if text:
            references.append(text)

    journal = _first_text_with_attribute(root, "title", "level", {"j"})
    doi = _first_text_with_attribute(root, "idno", "type", {"doi"})
    publication_date = ""
    for date in descendants(root, "date"):
        date_type = attribute_case_insensitive(date, "type").lower()
        if date_type in {"published", "publication"} or not publication_date:
            publication_date = attribute_case_insensitive(date, "when") or element_text(date)
        if date_type in {"published", "publication"}:
            break

    warnings = []
    if not body_sections:
        warnings.append("GROBID TEI did not contain extractable body paragraphs.")

    return ExtractedDocument(
        document_id=Path(source_identifier).stem,
        source_type="pdf",
        source_identifier=source_identifier,
        extraction_method="grobid_tei",
        metadata={
            "title": title,
            "authors": _tei_authors(root),
            "doi": doi,
            "journal": journal,
            "publication_date": publication_date,
        },
        sections=sections,
        references=references,
        diagnostics={
            "section_count": len(sections),
            "reference_count": len(references),
            "character_count": sum(len(section.text) for section in sections),
        },
        warnings=warnings,
        provenance={"requested_method": "grobid_tei", "fallback_used": False},
        raw_xml=tei_xml,
    )


class GrobidExtractor:
    """Extract PDFs by calling a separately managed GROBID service."""

    def __init__(
        self,
        base_url: str | None = None,
        connect_timeout: float = 5,
        read_timeout: float = 180,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("GROBID_URL") or "http://127.0.0.1:8070").rstrip("/")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.session = session or requests.Session()

    def is_alive(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}/api/isalive", timeout=self.connect_timeout
            )
            return response.ok
        except requests.RequestException:
            return False

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        pdf_path = Path(pdf_path).expanduser().resolve()
        if not pdf_path.is_file():
            raise GrobidError(f"PDF does not exist: {pdf_path}")
        if not self.is_alive():
            raise GrobidError(f"GROBID service is unavailable at {self.base_url}")

        try:
            with pdf_path.open("rb") as file_obj:
                response = self.session.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, file_obj, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "0",
                        "includeRawCitations": "0",
                    },
                    timeout=(self.connect_timeout, self.read_timeout),
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GrobidError(f"GROBID failed for {pdf_path.name}: {exc}") from exc

        document = parse_tei(response.text, str(pdf_path))
        document.provenance["grobid_url"] = self.base_url
        return document
