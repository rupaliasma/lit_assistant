import pytest

from litagent.tools import TOOLS, ToolBox, ToolError


def test_tool_schemas_are_strict():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_search_lists_papers(pubmed):
    box = ToolBox(pubmed)
    out = box.run("search_pubmed", {"query": "fourier primal dual MRI", "max_results": 5})
    assert out.splitlines()[0].startswith("40189690 | 2025 | Khatun R et al. Sci Rep 2025 | Complex-valued")
    assert box.searches == ["fourier primal dual MRI"]
    assert box.read == {}  # titles only: nothing counts as read yet


def test_evidence_requires_reading_first(pubmed):
    box = ToolBox(pubmed)
    with pytest.raises(ToolError, match="get_abstracts"):
        box.run("record_evidence", {"pmid": "40189690", "finding": "x"})
    out = box.run("get_abstracts", {"pmids": ["40189690", "99999999"]})
    assert "Abstract: Hyperthermia" in out and "PMID 99999999: not found" in out
    box.run("record_evidence", {"pmid": "40189690", "finding": "Temperature error fell to 0.6 C."})
    assert "[PMID:40189690] Khatun R et al." in box.evidence_block()
    assert "- Temperature error fell to 0.6 C." in box.evidence_block()


def test_full_text_paging(pubmed, monkeypatch):
    monkeypatch.setattr("litagent.tools.FULL_TEXT_PAGE", 40)
    box = ToolBox(pubmed)
    first = box.run("get_full_text", {"pmid": "40189690", "offset": 0})
    assert first.startswith("[characters 0-40 of") and "offset=40" in first
    assert "40189690" in box.read
    assert "No open-access full text" in box.run("get_full_text", {"pmid": "10542355", "offset": 0})


def test_bad_calls_become_tool_errors(pubmed):
    box = ToolBox(pubmed)
    with pytest.raises(ToolError):
        box.run("delete_everything", {})
    with pytest.raises(ToolError):
        box.run("get_abstracts", {"pmids": [str(i) for i in range(11)]})
    with pytest.raises(ToolError):
        box.run("get_abstracts", {"pmids": ["not-a-pmid"]})
