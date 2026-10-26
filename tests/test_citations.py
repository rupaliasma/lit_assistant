from litagent.citations import cited_pmids, extract_claims, split_sentences, uncited_statements

ANSWER = """SENSE unfolds aliased images using coil sensitivity maps [PMID:10542355]. GRAPPA instead works in k-space. \
[PMID:12111967] Both need calibration data [PMID: 12111967; PMID:23649942].
The literature on this point is sparse and evidence remains limited overall."""


def test_cited_pmids_in_order_without_duplicates():
    assert cited_pmids(ANSWER) == ["10542355", "12111967", "23649942"]
    assert cited_pmids("see [pmid 123, 456] and [PubMed: 789]") == ["123", "456", "789"]
    assert cited_pmids("no citations [1] here (2020)") == []


def test_citation_after_full_stop_belongs_to_previous_sentence():
    sentences = split_sentences(ANSWER)
    assert sentences[1] == "GRAPPA instead works in k-space. [PMID:12111967]"
    assert sentences[2].startswith("Both need calibration data")


def test_extract_claims():
    claims = extract_claims(ANSWER)
    assert [c.pmids for c in claims] == [["10542355"], ["12111967"], ["12111967", "23649942"]]


def test_uncited_statements_ignores_short_fragments_and_questions():
    text = "## Overview\nThis long sentence makes a claim without any support. Is that right for every case? Yes."
    assert uncited_statements(text) == ["This long sentence makes a claim without any support."]
    assert uncited_statements(ANSWER) == [
        "The literature on this point is sparse and evidence remains limited overall."
    ]
