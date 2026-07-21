from pathlib import Path

from final_project.extraction.pmc import PmcExtractor, parse_jats


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_jats_produces_common_schema_and_nested_sections():
    document = parse_jats(
        (FIXTURES / "pmc_sample.xml").read_text(encoding="utf-8"),
        "PMC123456",
    )

    assert document.extraction_method == "pmc_jats"
    assert document.metadata["pmcid"] == "PMC123456"
    assert document.metadata["pmid"] == "987654"
    assert document.metadata["doi"] == "10.1234/pmc-example"
    assert document.metadata["authors"] == ["Ana Example"]
    assert document.diagnostics["full_text_available"] is True
    assert document.diagnostics["body_character_count"] > 0
    assert "PMC body paragraph." in document.full_text()
    nested = next(section for section in document.sections if section.heading == "Nested methods")
    conclusion = next(section for section in document.sections if section.heading == "Conclusions")
    assert nested.level == 2
    assert nested.text == "Nested paragraph."
    assert conclusion.section_type == "conclusion"
    assert len(document.references) == 1


class FakeResponse:
    def __init__(self, *, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeNcbiSession:
    def __init__(self, jats_xml):
        self.headers = {}
        self.jats_xml = jats_xml
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        if url.endswith("esummary.fcgi"):
            return FakeResponse(
                payload={
                    "result": {
                        "987654": {
                            "articleids": [
                                {"idtype": "pmcid", "value": "PMC123456"}
                            ]
                        }
                    }
                }
            )
        if url.endswith("efetch.fcgi") and params["db"] == "pmc":
            return FakeResponse(text=self.jats_xml)
        raise AssertionError(f"Unexpected request: {url} {params}")


def test_pmc_extractor_resolves_pmid_without_network(monkeypatch):
    monkeypatch.setattr("final_project.extraction.pmc.time.sleep", lambda _delay: None)
    session = FakeNcbiSession(
        (FIXTURES / "pmc_sample.xml").read_text(encoding="utf-8")
    )
    extractor = PmcExtractor(email="researcher@example.org", session=session)

    document = extractor.extract(pmid="987654")

    assert document.document_id == "PMC123456"
    assert document.metadata["pmid"] == "987654"
    assert document.extraction_method == "pmc_jats"
    assert document.diagnostics["full_text_available"] is True
    assert len(session.calls) == 2


class AbstractOnlyNcbiSession:
    def __init__(self, pubmed_xml):
        self.headers = {}
        self.pubmed_xml = pubmed_xml

    def get(self, url, params, timeout):
        if url.endswith("esummary.fcgi"):
            return FakeResponse(payload={"result": {"987654": {"articleids": []}}})
        if url.endswith("efetch.fcgi") and params["db"] == "pubmed":
            return FakeResponse(text=self.pubmed_xml)
        raise AssertionError(f"Unexpected request: {url} {params}")


def test_pmid_requires_pmc_full_text_by_default(monkeypatch):
    import pytest

    from final_project.extraction.errors import PmcError

    monkeypatch.setattr("final_project.extraction.pmc.time.sleep", lambda _delay: None)
    session = AbstractOnlyNcbiSession("<PubmedArticleSet />")
    extractor = PmcExtractor(email="researcher@example.org", session=session)

    with pytest.raises(PmcError, match="No PMC full text"):
        extractor.extract(pmid="987654")
