"""PubMed Central full-text acquisition and JATS parsing."""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET

import requests

from .errors import PmcError
from .models import ExtractedDocument, Section
from .xml_utils import (
    attribute_case_insensitive,
    descendants,
    direct_children,
    element_text,
    first_descendant,
    paragraph_text,
)


EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _clean_doi(doi: str) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi.strip(), flags=re.I)


def _article_ids(article_meta: ET.Element | None) -> dict[str, str]:
    identifiers = {}
    for article_id in descendants(article_meta, "article-id"):
        id_type = attribute_case_insensitive(article_id, "pub-id-type").lower()
        if id_type:
            identifiers[id_type] = element_text(article_id)
    return identifiers


def _jats_authors(article_meta: ET.Element | None) -> list[str]:
    authors = []
    for contributor in descendants(article_meta, "contrib"):
        if attribute_case_insensitive(contributor, "contrib-type").lower() != "author":
            continue
        name = first_descendant(contributor, "name")
        surname = element_text(first_descendant(name, "surname"))
        given = element_text(first_descendant(name, "given-names"))
        full_name = " ".join(part for part in [given, surname] if part)
        if full_name and full_name not in authors:
            authors.append(full_name)
    return authors


def _publication_date(article_meta: ET.Element | None) -> str:
    pub_dates = list(descendants(article_meta, "pub-date"))
    if not pub_dates:
        return ""
    preferred = next(
        (
            item
            for item in pub_dates
            if attribute_case_insensitive(item, "pub-type").lower()
            in {"epub", "electronic", "ppub"}
        ),
        pub_dates[0],
    )
    year = element_text(first_descendant(preferred, "year"))
    month = element_text(first_descendant(preferred, "month"))
    day = element_text(first_descendant(preferred, "day"))
    return "-".join(part for part in [year, month.zfill(2), day.zfill(2)] if part)


def _jats_sections(body: ET.Element | None) -> list[Section]:
    if body is None:
        return []
    sections: list[Section] = []

    def add(heading: str, text: str, level: int) -> None:
        if not text:
            return
        kind = "conclusion" if "conclusion" in heading.lower() else "body"
        sections.append(Section(kind, text, heading, level, len(sections)))

    def walk(section: ET.Element, level: int) -> None:
        heading = element_text(next(iter(direct_children(section, "title")), None))
        add(heading, paragraph_text(direct_children(section, "p")), level)
        for child in direct_children(section, "sec"):
            walk(child, level + 1)

    add("", paragraph_text(direct_children(body, "p")), 1)
    for section in direct_children(body, "sec"):
        walk(section, 1)
    if not sections:
        add("", paragraph_text(descendants(body, "p")), 1)
    return sections


def parse_jats(jats_xml: str, requested_identifier: str) -> ExtractedDocument:
    """Parse a PMC JATS article into the normalized extraction schema."""
    try:
        root = ET.fromstring(jats_xml.encode("utf-8"))
    except ET.ParseError as exc:
        raise PmcError(f"PMC returned invalid JATS XML: {exc}") from exc

    front = first_descendant(root, "front")
    article_meta = first_descendant(front, "article-meta")
    journal_meta = first_descendant(front, "journal-meta")
    identifiers = _article_ids(article_meta)
    title = element_text(first_descendant(article_meta, "article-title"))
    abstract = paragraph_text(descendants(first_descendant(article_meta, "abstract"), "p"))
    body_sections = _jats_sections(first_descendant(root, "body"))

    sections: list[Section] = []
    if title:
        sections.append(Section("title", title, "", 0, len(sections)))
    if abstract:
        sections.append(Section("abstract", abstract, "Abstract", 1, len(sections)))
    for section in body_sections:
        section.order = len(sections)
        sections.append(section)

    references = []
    back = first_descendant(root, "back")
    for reference in descendants(first_descendant(back, "ref-list"), "ref"):
        text = element_text(reference)
        if text:
            references.append(text)

    pmcid = identifiers.get("pmc", requested_identifier).upper()
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    warnings = []
    if not body_sections:
        warnings.append("PMC article did not contain extractable body paragraphs.")
    body_character_count = sum(len(section.text) for section in body_sections)

    return ExtractedDocument(
        document_id=pmcid or requested_identifier,
        source_type="pmc",
        source_identifier=pmcid or requested_identifier,
        extraction_method="pmc_jats",
        metadata={
            "title": title,
            "authors": _jats_authors(article_meta),
            "doi": identifiers.get("doi", ""),
            "pmid": identifiers.get("pmid", ""),
            "pmcid": pmcid,
            "journal": element_text(first_descendant(journal_meta, "journal-title")),
            "publication_date": _publication_date(article_meta),
        },
        sections=sections,
        references=references,
        diagnostics={
            "section_count": len(sections),
            "reference_count": len(references),
            "character_count": sum(len(section.text) for section in sections),
            "body_character_count": body_character_count,
            "full_text_available": bool(body_sections),
        },
        warnings=warnings,
        provenance={"requested_method": "pmc_jats", "fallback_used": False},
        raw_xml=jats_xml,
    )


def parse_pubmed_abstract(pubmed_xml: str, pmid: str, doi: str = "") -> ExtractedDocument:
    """Parse an explicitly marked abstract-only PubMed fallback."""
    try:
        root = ET.fromstring(pubmed_xml.encode("utf-8"))
    except ET.ParseError as exc:
        raise PmcError(f"PubMed returned invalid XML: {exc}") from exc
    article = first_descendant(root, "Article")
    title = element_text(first_descendant(article, "ArticleTitle"))
    abstract = paragraph_text(descendants(first_descendant(article, "Abstract"), "AbstractText"))
    journal = element_text(first_descendant(first_descendant(article, "Journal"), "Title"))
    authors = []
    for author in descendants(first_descendant(article, "AuthorList"), "Author"):
        name = " ".join(
            part
            for part in [
                element_text(first_descendant(author, "ForeName")),
                element_text(first_descendant(author, "LastName")),
            ]
            if part
        )
        if name:
            authors.append(name)
    if not doi:
        for article_id in descendants(root, "ArticleId"):
            if attribute_case_insensitive(article_id, "IdType").lower() == "doi":
                doi = element_text(article_id)
                break
    sections = []
    if title:
        sections.append(Section("title", title, "", 0, len(sections)))
    if abstract:
        sections.append(Section("abstract", abstract, "Abstract", 1, len(sections)))
    return ExtractedDocument(
        document_id=f"PMID{pmid}",
        source_type="pubmed",
        source_identifier=pmid,
        extraction_method="pubmed_abstract",
        metadata={
            "title": title,
            "authors": authors,
            "doi": doi,
            "pmid": pmid,
            "pmcid": "",
            "journal": journal,
            "publication_date": "",
        },
        sections=sections,
        references=[],
        diagnostics={
            "section_count": len(sections),
            "reference_count": 0,
            "character_count": sum(len(section.text) for section in sections),
            "full_text_available": False,
        },
        warnings=["PMC full text was unavailable; this record contains only a PubMed abstract."],
        provenance={
            "requested_method": "pmc_jats",
            "fallback_used": True,
            "fallback_from": "pmc_jats",
            "fallback_reason": "No PMCID/full text was available.",
        },
        raw_xml=pubmed_xml,
    )


class PmcExtractor:
    """Resolve identifiers and fetch PMC full text through NCBI E-utilities."""

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 60,
    ) -> None:
        self.email = email or os.getenv("NCBI_EMAIL", "")
        self.api_key = api_key or os.getenv("NCBI_API_KEY", "")
        if not self.email:
            raise PmcError("NCBI_EMAIL is required for PMC/PubMed requests")
        self.session = session or requests.Session()
        self.timeout = timeout
        self._last_request_at = 0.0
        self.session.headers.setdefault(
            "User-Agent", f"final-project-rag/0.1 ({self.email})"
        )

    def _request(self, endpoint: str, params: dict[str, str]) -> requests.Response:
        minimum_delay = 0.11 if self.api_key else 0.34
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < minimum_delay:
            time.sleep(minimum_delay - elapsed)
        request_params = {**params, "email": self.email}
        if self.api_key:
            request_params["api_key"] = self.api_key
        try:
            response = self.session.get(
                f"{EUTILS_URL}/{endpoint}",
                params=request_params,
                timeout=self.timeout,
            )
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise PmcError(f"NCBI request failed for {endpoint}: {exc}") from exc

    def doi_to_pmid(self, doi: str) -> str | None:
        response = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": f"{_clean_doi(doi)}[DOI]",
                "retmode": "json",
                "retmax": "1",
            },
        )
        identifiers = response.json().get("esearchresult", {}).get("idlist", [])
        return identifiers[0] if identifiers else None

    def pmid_to_pmcid(self, pmid: str) -> str | None:
        response = self._request(
            "esummary.fcgi",
            {"db": "pubmed", "id": str(pmid), "retmode": "json"},
        )
        payload = response.json().get("result", {})
        record = payload.get(str(pmid))
        if record is None and payload.get("uids"):
            record = payload.get(str(payload["uids"][0]), {})
        identifiers = (record or {}).get("articleids", [])
        for identifier in identifiers:
            if identifier.get("idtype") == "pmcid":
                value = str(identifier.get("value", "")).replace("pmc-id:", "").replace(";", "").strip()
                return value.upper() or None
        return None

    def fetch_pmc_xml(self, pmcid: str) -> str:
        return self._request(
            "efetch.fcgi",
            {"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"},
        ).text

    def fetch_pubmed_xml(self, pmid: str) -> str:
        return self._request(
            "efetch.fcgi",
            {"db": "pubmed", "id": pmid, "retmode": "xml"},
        ).text

    def extract(
        self,
        *,
        pmcid: str | None = None,
        pmid: str | None = None,
        doi: str | None = None,
        allow_abstract_fallback: bool = False,
    ) -> ExtractedDocument:
        if not any((pmcid, pmid, doi)):
            raise PmcError("Provide one of pmcid, pmid, or doi")
        normalized_doi = _clean_doi(doi) if doi else ""
        if not pmid and doi:
            pmid = self.doi_to_pmid(doi)
            if not pmid:
                raise PmcError(f"No PubMed record found for DOI {normalized_doi}")
        if not pmcid and pmid:
            pmcid = self.pmid_to_pmcid(str(pmid))

        if pmcid:
            normalized_pmcid = pmcid.upper()
            if not normalized_pmcid.startswith("PMC"):
                normalized_pmcid = f"PMC{normalized_pmcid}"
            document = parse_jats(
                self.fetch_pmc_xml(normalized_pmcid), normalized_pmcid
            )
            if not document.diagnostics["full_text_available"]:
                raise PmcError(
                    f"PMC record {normalized_pmcid} did not contain extractable body text"
                )
            if pmid and not document.metadata.get("pmid"):
                document.metadata["pmid"] = str(pmid)
            if normalized_doi and not document.metadata.get("doi"):
                document.metadata["doi"] = normalized_doi
            document.provenance["ncbi_email"] = self.email
            return document

        if pmid and allow_abstract_fallback:
            document = parse_pubmed_abstract(
                self.fetch_pubmed_xml(str(pmid)), str(pmid), normalized_doi
            )
            document.provenance["ncbi_email"] = self.email
            return document
        raise PmcError(f"No PMC full text is available for PMID {pmid}")
