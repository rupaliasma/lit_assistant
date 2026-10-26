import pytest

from conftest import json_reply, reply, text_block, tool_block
from litagent.agents import LiteratureAssistant, check_citations, run_researcher
from litagent.llm import FALLBACK_BETA, RefusalError
from litagent.tools import ToolBox

RESEARCH = [
    reply(tool_block("search_pubmed", {"query": "Fourier PD thermometry", "max_results": 5})),
    # Recording before reading is refused; the agent gets an error result and carries on.
    reply(tool_block("record_evidence", {"pmid": "40189690", "finding": "too early"}, id="tu_early")),
    reply(tool_block("get_abstracts", {"pmids": ["40189690"]})),
    reply(tool_block("record_evidence", {"pmid": "40189690", "finding": "Error fell from 1.3 to 0.6 C at R=10."})),
    reply(text_block("One key paper covers the question.")),
]
DRAFT = (
    "Fourier-PD reduced the temperature error from 1.3 to 0.6 degrees at tenfold acceleration [PMID:40189690]. "
    "SENSE was used as the baseline in that study [PMID:10542355]."
)
FINAL = "Fourier-PD reduced the temperature error from 1.3 to 0.6 degrees at tenfold acceleration [PMID:40189690]."


def supported(claim=0):
    return json_reply(
        {"verdicts": [{"claim": claim, "pmid": "40189690", "verdict": "supported", "reason": "abstract"}]}
    )


def test_pipeline_revises_unverifiable_citations(fake_llm, pubmed):
    llm = fake_llm(*RESEARCH, reply(text_block(DRAFT)), supported(), reply(text_block(FINAL)), supported())
    result = LiteratureAssistant(llm=llm, pubmed=pubmed).ask("How accurate is Fourier-PD thermometry?")

    assert result.revised and result.answer == FINAL
    assert [(c.pmid, c.status) for c in result.checks] == [("40189690", "supported")]
    assert result.citation_precision == 1.0
    assert [s["pmid"] for s in result.sources] == ["40189690"]
    assert result.papers_read == ["40189690"]
    assert result.usage["calls"] == 9 and result.usage["input_tokens"] == 900

    requests = llm.client.requests
    early = requests[2]["messages"][-1]["content"][0]
    assert early["tool_use_id"] == "tu_early" and early["is_error"]
    # The revision prompt tells the writer what was wrong with the draft.
    assert "not_in_evidence" in requests[7]["messages"][0]["content"]


def test_no_revision_when_everything_checks_out(fake_llm, pubmed):
    llm = fake_llm(*RESEARCH, reply(text_block(FINAL)), supported())
    result = LiteratureAssistant(llm=llm, pubmed=pubmed).ask("How accurate is Fourier-PD thermometry?")
    assert not result.revised and result.usage["calls"] == 7


def test_checker_flags_missing_and_skipped_pairs(fake_llm, pubmed):
    box = ToolBox(pubmed)
    box.run("get_abstracts", {"pmids": ["40189690"]})
    box.run("record_evidence", {"pmid": "40189690", "finding": "f"})
    answer = "A first claim is made here [PMID:40189690]. A second claim cites a ghost paper [PMID:99999999]."
    checks = check_citations(answer, box, pubmed, fake_llm(json_reply({"verdicts": []})))
    assert [(c.pmid, c.status) for c in checks] == [("40189690", "unsupported"), ("99999999", "not_found")]


def test_researcher_stops_after_max_turns(fake_llm, pubmed):
    loop = [reply(tool_block("search_pubmed", {"query": "q", "max_results": 1}, id=f"t{i}")) for i in range(3)]
    notes = run_researcher("q", ToolBox(pubmed), fake_llm(*loop), max_turns=3)
    assert "stopped after 3 turns" in notes


def test_llm_request_shape(fake_llm):
    llm = fake_llm(json_reply({"ok": True}))
    assert llm.json(system="s", prompt="p", schema={"type": "object"}) == {"ok": True}
    req = llm.client.requests[0]
    assert req["model"] == "test-model" and req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert "temperature" not in req and "thinking" not in req


def test_fallbacks_can_be_disabled(fake_llm):
    llm = fake_llm(reply(text_block("hi")), fallbacks=False)
    llm.create(system="s", messages=[{"role": "user", "content": "x"}])
    assert "fallbacks" not in llm.client.requests[0] and "betas" not in llm.client.requests[0]


def test_refusal_raises(fake_llm):
    llm = fake_llm(reply(stop_reason="refusal"))
    with pytest.raises(RefusalError):
        llm.create(system="s", messages=[{"role": "user", "content": "x"}])


def test_truncated_structured_output_raises(fake_llm):
    llm = fake_llm(reply(text_block('{"verd'), stop_reason="max_tokens"))
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm.json(system="s", prompt="p", schema={})
