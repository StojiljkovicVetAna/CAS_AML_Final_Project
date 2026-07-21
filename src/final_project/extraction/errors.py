"""Extraction-specific exceptions."""


class ExtractionError(RuntimeError):
    """Base error for an extraction operation."""


class GrobidError(ExtractionError):
    """GROBID was unavailable or could not process a PDF."""


class PmcError(ExtractionError):
    """PMC/PubMed lookup, download, or parsing failed."""


class PdfExtractionError(ExtractionError):
    """The local PDF fallback could not extract a document."""
