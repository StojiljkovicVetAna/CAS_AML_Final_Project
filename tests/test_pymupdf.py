from final_project.extraction.pymupdf import remove_repeated_page_lines, sections_from_text


def test_repeated_lines_are_counted_per_page():
    pages = [
        "Journal Header\nFirst page text",
        "Journal Header\nSecond page text\nSecond page text",
        "Journal Header\nThird page text",
    ]
    cleaned, removed = remove_repeated_page_lines(pages, threshold=0.7)

    assert removed == ["Journal Header"]
    assert all("Journal Header" not in page for page in cleaned)
    assert "Second page text" not in removed


def test_heading_is_metadata_not_prose():
    title, sections, _ = sections_from_text(
        "A Paper Title\nABSTRACT\nAbstract sentence.\n"
        "1 Introduction\nIntroduction sentence.\n"
        "CONCLUSION\nConclusion sentence."
    )

    introduction = next(section for section in sections if section.heading == "1 Introduction")
    conclusion = next(section for section in sections if section.section_type == "conclusion")
    assert "1 Introduction" not in introduction.text
    assert introduction.text == "Introduction sentence."
    assert conclusion.heading == "CONCLUSION"
    assert title
