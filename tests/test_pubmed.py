import httpx
import pytest

from conftest import FIXTURES
from litagent.pubmed import PubMed, parse_efetch, parse_jats_body


def test_parse_efetch_fields():
    papers = {p.pmid: p for p in parse_efetch((FIXTURES / "efetch.xml").read_text())}
    khatun = papers["40189690"]
    assert khatun.title.startswith("Complex-valued neural networks to speed-up MR thermometry")
    assert khatun.journal == "Sci Rep" and khatun.year == "2025"
    assert khatun.authors[:2] == ["Khatun R", "Chatterjee S"]
    assert khatun.doi == "10.1038/s41598-025-96071-x"
    assert khatun.pmcid == "PMC11973158"
    assert "hyperthermia" in khatun.abstract.lower()
    assert khatun.url == "https://pubmed.ncbi.nlm.nih.gov/40189690/"
    assert khatun.short_ref() == "Khatun R et al. Sci Rep 2025"


def test_reference_list_ids_do_not_leak_into_the_article():
    sense = {p.pmid: p for p in parse_efetch((FIXTURES / "efetch.xml").read_text())}["10542355"]
    # The SENSE paper has no PMC copy or DOI of its own; IDs in its reference list must not be picked up.
    assert sense.pmcid is None and sense.doi is None
    assert sense.title == "SENSE: sensitivity encoding for fast MRI."


def test_search_and_fetch(pubmed):
    assert pubmed.search("fourier primal dual MRI", 5)[0] == "40189690"
    papers = pubmed.fetch(["10542355", "40189690"])
    assert [p.pmid for p in papers] == ["10542355", "40189690"]
    assert pubmed.exists("40189690") and not pubmed.exists("99999999") and not pubmed.exists("abc")


def test_fetch_rejects_non_pmids(pubmed):
    with pytest.raises(ValueError):
        pubmed.fetch(["10.1038/s41598"])


def test_requests_identify_the_tool():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    pm = PubMed(http=httpx.Client(transport=httpx.MockTransport(handler)), api_key="k", email="me@example.org")
    pm.search("x")
    assert seen[0]["tool"] == "litagent" and seen[0]["api_key"] == "k" and seen[0]["email"] == "me@example.org"


def test_retries_rate_limited_requests(monkeypatch):
    monkeypatch.setattr("litagent.pubmed.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})

    pm = PubMed(http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert pm.search("x") == ["1"] and len(calls) == 2


def test_full_text(pubmed):
    text = pubmed.full_text("40189690")
    assert text.startswith("## Introduction")
    assert "## Thermometry\nThe PRF shift method" in text
    assert "Should not appear" not in text
    assert pubmed.full_text("10542355") is None  # not in PMC


def test_jats_without_body():
    assert parse_jats_body("<article><front/></article>") == ""
