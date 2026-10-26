import json
from pathlib import Path

import pytest

from conftest import json_reply
from litagent.evaluate import (
    evaluate,
    format_summary,
    gold_recall,
    grade_citations,
    grade_correctness,
    load_questions,
)
from test_api import StubAssistant

QUESTIONS = Path(__file__).parents[1] / "data" / "questions.jsonl"


def test_question_set_is_well_formed():
    questions = load_questions(QUESTIONS)
    assert len(questions) >= 20
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert q["question"].endswith("?") and len(q["key_points"]) >= 3
        assert q["gold_pmids"] and all(p.isdigit() for p in q["gold_pmids"])


def test_grade_correctness(fake_llm):
    judge = fake_llm(
        json_reply(
            {
                "points": [
                    {"index": 0, "grade": "covered"},
                    {"index": 1, "grade": "partial"},
                    {"index": 2, "grade": "contradicted"},
                ]
            }
        )
    )
    result = grade_correctness("answer", ["a", "b", "c", "d"], judge)  # point 3 not graded -> missing
    assert result["correctness"] == 1.5 / 4
    assert result["contradictions"] == 1
    assert result["point_grades"] == ["covered", "partial", "contradicted", "missing"]


def test_grade_citations_counts_ghost_pmids_as_unsupported(fake_llm, pubmed):
    answer = "A claim backed by the paper [PMID:40189690]. Another claim citing nothing real [PMID:99999999]."
    judge = fake_llm(json_reply({"pairs": [{"index": 0, "supported": "yes"}]}))
    result = grade_citations(answer, pubmed, judge)
    assert result == {"citation_validity": 0.5, "citation_accuracy": 0.5, "n_citation_pairs": 2}
    assert grade_citations("No citations at all.", pubmed, judge)["citation_accuracy"] is None


def test_gold_recall():
    assert gold_recall(["1", "2"], cited={"1", "3"}, read={"1", "2"}) == {
        "gold_recall_cited": 0.5,
        "gold_recall_read": 1.0,
    }


def test_evaluate_end_to_end(fake_llm, pubmed, tmp_path):
    questions = [
        {"id": "grappa", "question": "How does GRAPPA work?", "key_points": ["k-space"], "gold_pmids": ["12111967"]},
        {"id": "boom", "question": "This one fails", "key_points": ["x"], "gold_pmids": []},
    ]

    class Flaky(StubAssistant):
        def ask(self, question):
            if "fails" in question:
                raise RuntimeError("network down")
            return super().ask(question)

    assistant = Flaky()
    assistant.pubmed, assistant.llm = pubmed, fake_llm()
    judge = fake_llm(json_reply({"points": [{"index": 0, "grade": "covered"}]}))
    summary = evaluate(questions, assistant, judge, tmp_path)

    assert summary["questions"] == 2 and summary["failed"] == 1
    assert summary["correctness"]["mean"] == 1.0
    assert summary["gold_recall_cited"]["mean"] == 1.0
    # 12111967 is not in the offline fixture, so it counts as an invalid citation.
    assert summary["citation_validity"]["mean"] == 0.0
    lines = (tmp_path / "answers.jsonl").read_text().splitlines()
    assert json.loads(lines[1])["error"] == "RuntimeError('network down')"
    assert "| correctness | 1.000 | 1 |" in format_summary(summary)


def test_malformed_judge_output_is_an_error_not_a_zero(fake_llm):
    with pytest.raises(ValueError, match="does not match the schema"):
        grade_correctness("answer", ["a"], fake_llm(json_reply({"grades": "all good"})))
