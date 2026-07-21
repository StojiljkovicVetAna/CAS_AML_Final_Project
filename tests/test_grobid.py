from pathlib import Path

from final_project.extraction.grobid import parse_tei


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_tei_preserves_structure_without_heading_in_text():
    document = parse_tei(
        (FIXTURES / "grobid_sample.tei.xml").read_text(encoding="utf-8"),
        "paper.pdf",
    )

    assert document.extraction_method == "grobid_tei"
    assert document.metadata["title"] == "Structured Dog Study"
    assert document.metadata["authors"] == ["Ana Example"]
    assert document.metadata["journal"] == "Journal of Examples"
    introduction = next(section for section in document.sections if section.heading == "Introduction")
    conclusion = next(section for section in document.sections if section.heading == "Conclusion")
    assert introduction.text == "First body paragraph."
    assert "Introduction" not in introduction.text
    assert conclusion.section_type == "conclusion"
    assert len(document.references) == 1
