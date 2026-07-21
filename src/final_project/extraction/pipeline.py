"""High-level extraction routing and fallback policy."""

from __future__ import annotations

from pathlib import Path

from .errors import GrobidError, PdfExtractionError
from .grobid import GrobidExtractor
from .models import ExtractedDocument
from .pmc import PmcExtractor
from .pymupdf import PyMuPDFExtractor


class ExtractionPipeline:
    """Route paper sources through the configured extraction methods."""

    def __init__(
        self,
        grobid_extractor: GrobidExtractor | None = None,
        pdf_fallback: PyMuPDFExtractor | None = None,
        pmc_extractor: PmcExtractor | None = None,
        *,
        enable_pdf_fallback: bool = True,
        minimum_grobid_body_characters: int = 200,
    ) -> None:
        self.grobid_extractor = grobid_extractor or GrobidExtractor()
        self.pdf_fallback = pdf_fallback or PyMuPDFExtractor()
        self.pmc_extractor = pmc_extractor
        self.enable_pdf_fallback = enable_pdf_fallback
        self.minimum_grobid_body_characters = minimum_grobid_body_characters

    def _validate_grobid_output(self, document: ExtractedDocument) -> None:
        body_characters = sum(
            len(section.text)
            for section in document.sections
            if section.section_type in {"body", "conclusion"}
        )
        if body_characters < self.minimum_grobid_body_characters:
            raise GrobidError(
                "GROBID output did not contain enough body text "
                f"({body_characters} characters)"
            )

    def extract_pdf(self, pdf_path: str | Path) -> ExtractedDocument:
        """Use GROBID by default and cleaned PyMuPDF only as fallback."""
        try:
            document = self.grobid_extractor.extract(pdf_path)
            self._validate_grobid_output(document)
            return document
        except GrobidError as grobid_error:
            if not self.enable_pdf_fallback:
                raise
            try:
                document = self.pdf_fallback.extract(pdf_path)
            except PdfExtractionError as fallback_error:
                raise PdfExtractionError(
                    f"GROBID failed ({grobid_error}); PyMuPDF fallback also failed "
                    f"({fallback_error})"
                ) from fallback_error
            document.provenance.update(
                {
                    "requested_method": "grobid_tei",
                    "fallback_used": True,
                    "fallback_from": "grobid_tei",
                    "fallback_reason": str(grobid_error),
                }
            )
            document.warnings.insert(
                0, f"GROBID was not used; cleaned PyMuPDF fallback: {grobid_error}"
            )
            return document

    def extract_online(
        self,
        *,
        pmcid: str | None = None,
        pmid: str | None = None,
        doi: str | None = None,
        allow_abstract_fallback: bool = False,
        email: str | None = None,
        ncbi_api_key: str | None = None,
    ) -> ExtractedDocument:
        """Extract PMC full text, resolving PMID/DOI when necessary."""
        extractor = self.pmc_extractor
        if extractor is None:
            extractor = PmcExtractor(email=email, api_key=ncbi_api_key)
        return extractor.extract(
            pmcid=pmcid,
            pmid=pmid,
            doi=doi,
            allow_abstract_fallback=allow_abstract_fallback,
        )
