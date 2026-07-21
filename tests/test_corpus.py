import json
from pathlib import Path

from final_project.extraction.corpus import normalize_doi, read_doi_list, run_corpus
from final_project.extraction.models import ExtractedDocument, Section


def make_document(document_id: str, source_type: str, source: str, doi: str):
    return ExtractedDocument(
        document_id=document_id,
        source_type=source_type,
        source_identifier=source,
        extraction_method="grobid_tei" if source_type == "pdf" else "pmc_jats",
        metadata={"title": document_id, "doi": doi},
        sections=[Section("body", "A complete body paragraph.")],
        diagnostics={"character_count": 26},
    )


class FakePipeline:
    def __init__(self, pdf_doi="10.1000/same"):
        self.pdf_doi = pdf_doi
        self.pdf_calls = []
        self.online_calls = []

    def extract_pdf(self, path):
        self.pdf_calls.append(Path(path))
        return make_document(Path(path).stem, "pdf", str(Path(path).resolve()), self.pdf_doi)

    def extract_online(self, *, doi, allow_abstract_fallback):
        self.online_calls.append((doi, allow_abstract_fallback))
        return make_document("PMC-online", "pmc", doi, doi)


def test_normalize_and_read_dois(tmp_path):
    doi_file = tmp_path / "list_DOIs.txt"
    doi_file.write_text(
        "https://doi.org/10.1000/ABC\n10.1000/abc\nnot-a-doi\n", encoding="utf-8"
    )

    dois, failures = read_doi_list(doi_file)

    assert normalize_doi("DOI: 10.1000/ABC.") == "10.1000/abc"
    assert dois == ["10.1000/abc"]
    assert len(failures) == 1


def test_corpus_skips_doi_represented_by_pdf_and_is_incremental(tmp_path):
    pdf_dir = tmp_path / "input" / "PDFs"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "paper.pdf"
    pdf_path.write_bytes(b"first PDF version")
    doi_file = tmp_path / "input" / "list_DOIs.txt"
    doi_file.write_text("https://doi.org/10.1000/same\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    first_pipeline = FakePipeline()
    first = run_corpus(
        first_pipeline, pdf_dir=pdf_dir, doi_list=doi_file, output_dir=output_dir
    )

    assert len(first_pipeline.pdf_calls) == 1
    assert first_pipeline.online_calls == []
    assert [item["reason"] for item in first.skipped] == ["doi_present_in_pdf"]
    assert (output_dir / "documents" / "paper.json").is_file()

    second_pipeline = FakePipeline()
    second = run_corpus(
        second_pipeline, pdf_dir=pdf_dir, doi_list=doi_file, output_dir=output_dir
    )

    assert second_pipeline.pdf_calls == []
    assert second_pipeline.online_calls == []
    assert {item["reason"] for item in second.skipped} == {
        "unchanged_pdf",
        "already_extracted",
    }
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["document_count"] == 1
    assert manifest["last_run"]["num_processed"] == 0


def test_corpus_processes_new_doi_and_changed_pdf(tmp_path):
    pdf_dir = tmp_path / "PDFs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper.pdf"
    pdf_path.write_bytes(b"version one")
    doi_file = tmp_path / "list_DOIs.txt"
    doi_file.write_text("10.1000/online\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    first_pipeline = FakePipeline(pdf_doi="10.1000/pdf")
    run_corpus(first_pipeline, pdf_dir=pdf_dir, doi_list=doi_file, output_dir=output_dir)
    assert len(first_pipeline.pdf_calls) == 1
    assert first_pipeline.online_calls == [("10.1000/online", False)]

    pdf_path.write_bytes(b"version two")
    second_pipeline = FakePipeline(pdf_doi="10.1000/pdf")
    second = run_corpus(
        second_pipeline, pdf_dir=pdf_dir, doi_list=doi_file, output_dir=output_dir
    )

    assert len(second_pipeline.pdf_calls) == 1
    assert second_pipeline.online_calls == []
    assert len(second.processed) == 1


def test_corpus_keeps_one_document_for_two_pdfs_with_same_doi(tmp_path):
    pdf_dir = tmp_path / "PDFs"
    pdf_dir.mkdir()
    (pdf_dir / "first.pdf").write_bytes(b"first")
    (pdf_dir / "second.pdf").write_bytes(b"second")
    doi_file = tmp_path / "list_DOIs.txt"
    doi_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "output"

    pipeline = FakePipeline(pdf_doi="10.1000/duplicate")
    result = run_corpus(
        pipeline, pdf_dir=pdf_dir, doi_list=doi_file, output_dir=output_dir
    )

    assert len(pipeline.pdf_calls) == 2
    assert len(result.processed) == 1
    assert result.skipped[0]["reason"] == "doi_already_in_corpus"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["document_count"] == 1
