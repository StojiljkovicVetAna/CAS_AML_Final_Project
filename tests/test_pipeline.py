import pytest

from final_project.extraction.errors import GrobidError
from final_project.extraction.models import ExtractedDocument, Section
from final_project.extraction.pipeline import ExtractionPipeline


class FailingGrobid:
    def extract(self, _path):
        raise GrobidError("service unavailable")


class SuccessfulFallback:
    def extract(self, path):
        return ExtractedDocument(
            document_id="paper",
            source_type="pdf",
            source_identifier=str(path),
            extraction_method="pymupdf_cleaned",
            sections=[Section("body", "Fallback body text.")],
        )


def test_pipeline_records_pdf_fallback_reason():
    pipeline = ExtractionPipeline(
        grobid_extractor=FailingGrobid(),
        pdf_fallback=SuccessfulFallback(),
    )
    document = pipeline.extract_pdf("paper.pdf")

    assert document.extraction_method == "pymupdf_cleaned"
    assert document.provenance["fallback_used"] is True
    assert document.provenance["fallback_from"] == "grobid_tei"
    assert "service unavailable" in document.provenance["fallback_reason"]


def test_pipeline_can_disable_pdf_fallback():
    pipeline = ExtractionPipeline(
        grobid_extractor=FailingGrobid(),
        pdf_fallback=SuccessfulFallback(),
        enable_pdf_fallback=False,
    )
    with pytest.raises(GrobidError):
        pipeline.extract_pdf("paper.pdf")
