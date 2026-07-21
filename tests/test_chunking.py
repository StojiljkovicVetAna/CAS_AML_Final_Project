import json

from final_project.chunking import ChunkConfig, chunk_document
from final_project.chunking.pipeline import chunk_document_dir
from final_project.chunking.text import split_sentences


def _document():
    return {
        "schema_version": "1.0",
        "document_id": "paper-1",
        "source_type": "pdf",
        "source_identifier": "paper.pdf",
        "extraction_method": "grobid_tei",
        "metadata": {
            "title": "A test paper",
            "doi": "10.123/example",
            "journal": "Journal of Tests",
        },
        "sections": [
            {
                "section_type": "title",
                "heading": "",
                "text": "A test paper",
                "order": 0,
            },
            {
                "section_type": "abstract",
                "heading": "Abstract",
                "text": (
                    "Dogs followed the actor. Horn et al. reported a similar effect. "
                    "The online version contains supplementary material available at 10.123/example."
                ),
                "order": 1,
            },
            {
                "section_type": "body",
                "heading": "Results",
                "text": (
                    "Fig. 1 Relationship between gaze and condition.\n\n"
                    "Dogs looked longer at the positive actor. "
                    "This effect was visible in condition A. "
                    "Table 2 Summary statistics.\n\n"
                    "Measured responses were stable across trials. "
                    "The result supports the main hypothesis."
                ),
                "order": 2,
            },
            {
                "section_type": "body",
                "heading": "References",
                "text": "Smith A. Example reference.",
                "order": 3,
            },
        ],
        "references": ["Smith A. Example reference."],
        "provenance": {},
    }


def test_sentence_splitter_keeps_academic_abbreviations():
    sentences = split_sentences(
        "Horn et al. reported the effect. Fig. 1 shows the design. The result was clear."
    )
    assert sentences == [
        "Horn et al. reported the effect.",
        "Fig. 1 shows the design.",
        "The result was clear.",
    ]


def test_chunk_document_filters_captions_notices_and_references():
    chunks = chunk_document(
        _document(),
        config=ChunkConfig(target_tokens=80, max_tokens=120, overlap_tokens=20, min_tokens=10),
    )
    combined = "\n".join(chunk.text for chunk in chunks)

    assert chunks
    assert all(chunk.section_type != "title" for chunk in chunks)
    assert all(chunk.heading != "References" for chunk in chunks)
    assert "Horn et al. reported a similar effect." in combined
    assert "Fig. 1 Relationship" not in combined
    assert "Table 2 Summary" not in combined
    assert "supplementary material" not in combined.lower()
    assert "Smith A. Example reference." not in combined
    assert chunks[0].metadata["doi"] == "10.123/example"


def test_chunk_document_dir_writes_jsonl_and_manifest(tmp_path):
    document_dir = tmp_path / "documents"
    document_dir.mkdir()
    (document_dir / "paper-1.json").write_text(json.dumps(_document()), encoding="utf-8")

    manifest = chunk_document_dir(
        document_dir,
        output_dir=tmp_path / "chunks",
        config=ChunkConfig(target_tokens=80, max_tokens=120, overlap_tokens=20, min_tokens=10),
    )

    assert manifest["num_documents"] == 1
    assert manifest["num_chunks"] > 0
    chunks_path = tmp_path / "chunks" / "chunks.jsonl"
    manifest_path = tmp_path / "chunks" / "chunk_manifest.json"
    assert chunks_path.exists()
    assert manifest_path.exists()
    first_chunk = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_chunk["document_id"] == "paper-1"
    assert first_chunk["provenance"]["chunker"] == "section_aware_academic_v1"


def test_oversized_sentence_is_split_before_chunking():
    document = _document()
    long_sentence = " ".join(f"token{i}" for i in range(260)) + "."
    document["sections"] = [
        {
            "section_type": "body",
            "heading": "Methods",
            "text": long_sentence,
            "order": 1,
        }
    ]

    chunks = chunk_document(
        document,
        config=ChunkConfig(target_tokens=80, max_tokens=120, overlap_tokens=20, min_tokens=10),
    )

    assert len(chunks) > 1
    assert max(chunk.token_count for chunk in chunks) <= 120
