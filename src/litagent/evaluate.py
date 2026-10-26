"""Evaluation on the hand-curated question set (``data/questions.jsonl``).

Each question has key points a good answer should make and the PMIDs of the
papers that establish them. For every answer we report

* correctness: share of key points the answer covers (partial = 0.5), judged by
  an LLM grader that sees only the answer and the key points; contradictions are
  counted separately;
* citation validity: share of cited PMIDs that exist in PubMed;
* citation accuracy: share of claim/citation pairs whose cited abstract supports
  the claim, judged by an independent grader prompt (not the pipeline's own
  checker, which could be lenient with itself);
* gold recall: share of the curated key papers that were cited, and that were
  at least read during research.
"""

import json
import logging
import statistics
import time
from pathlib import Path

from litagent.agents import LiteratureAssistant
from litagent.citations import extract_claims

log = logging.getLogger(__name__)

CORRECTNESS_SYSTEM = """\
You grade answers to questions about MRI reconstruction against a list of key points written by \
an expert. For each key point decide whether the answer "covered" it (states it or something \
equivalent), "partial" (touches on it but incompletely or vaguely), "missing", or "contradicted" \
(says something incompatible with it). Judge meaning, not wording."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "grade": {"type": "string", "enum": ["covered", "partial", "missing", "contradicted"]},
                },
                "required": ["index", "grade"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["points"],
    "additionalProperties": False,
}

SUPPORT_SYSTEM = """\
You are an independent reviewer auditing citations. For each numbered pair you get a sentence \
and the title and abstract of the paper cited for it. Answer "yes" if the abstract supports the \
sentence, "partly" if it supports only part of it, and "no" otherwise. Use only the abstract."""

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "string", "enum": ["yes", "partly", "no"]},
                },
                "required": ["index", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


def _field(result, key: str) -> list:
    """The list the grader schema asks for; malformed judge output is an error, not a zero score."""
    if not isinstance(result, dict) or not isinstance(result.get(key), list):
        raise ValueError(f"judge output does not match the schema (no {key!r} list); use a more capable judge model")
    return result[key]


POINT_SCORE = {"covered": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
SUPPORT_SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}


def load_questions(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def grade_correctness(answer: str, key_points: list[str], judge) -> dict:
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(key_points))
    result = judge.json(
        system=CORRECTNESS_SYSTEM, prompt=f"Key points:\n{points}\n\nAnswer:\n{answer}", schema=CORRECTNESS_SCHEMA
    )
    grades = {p.get("index"): p.get("grade") for p in _field(result, "points") if isinstance(p, dict)}
    graded = [grades.get(i) if grades.get(i) in POINT_SCORE else "missing" for i in range(len(key_points))]
    return {
        "correctness": statistics.fmean(POINT_SCORE[g] for g in graded),
        "contradictions": graded.count("contradicted"),
        "point_grades": graded,
    }


def grade_citations(answer: str, pubmed, judge) -> dict:
    pairs = [(c.text, pmid) for c in extract_claims(answer) for pmid in c.pmids]
    if not pairs:
        return {"citation_validity": None, "citation_accuracy": None, "n_citation_pairs": 0}
    papers = {p.pmid: p for p in pubmed.fetch(sorted({pmid for _, pmid in pairs}))}
    valid = [(text, pmid) for text, pmid in pairs if pmid in papers]

    scores = []
    if valid:
        listing = "\n\n".join(
            f"{i}. Sentence: {text}\n   Paper: {papers[pmid].title}\n   Abstract: {papers[pmid].abstract}"
            for i, (text, pmid) in enumerate(valid)
        )
        result = judge.json(system=SUPPORT_SYSTEM, prompt=listing, schema=SUPPORT_SCHEMA)
        verdicts = {p.get("index"): p.get("supported") for p in _field(result, "pairs") if isinstance(p, dict)}
        scores = [SUPPORT_SCORE.get(verdicts.get(i), 0.0) for i in range(len(valid))]
    # Pairs citing non-existent PMIDs count as unsupported.
    return {
        "citation_validity": len(valid) / len(pairs),
        "citation_accuracy": sum(scores) / len(pairs),
        "n_citation_pairs": len(pairs),
    }


def gold_recall(gold: list[str], cited: set[str], read: set[str]) -> dict:
    if not gold:
        return {"gold_recall_cited": None, "gold_recall_read": None}
    return {
        "gold_recall_cited": len(set(gold) & cited) / len(gold),
        "gold_recall_read": len(set(gold) & read) / len(gold),
    }


def evaluate(questions: list[dict], assistant: LiteratureAssistant, judge, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "answers.jsonl").open("w") as fh:
        for q in questions:
            log.info("[%s] %s", q["id"], q["question"])
            tic = time.time()
            try:
                answer = assistant.ask(q["question"])
                cited = {c.pmid for c in answer.checks}
                record = {
                    "id": q["id"],
                    "seconds": round(time.time() - tic, 1),
                    **grade_correctness(answer.answer, q["key_points"], judge),
                    **grade_citations(answer.answer, assistant.pubmed, judge),
                    **gold_recall(q.get("gold_pmids", []), cited, set(answer.papers_read)),
                    "self_check_precision": answer.citation_precision,
                    "answer": answer.to_dict(),
                }
            except Exception as err:  # keep going: one failure should not sink the whole run
                log.exception("question %s failed", q["id"])
                record = {"id": q["id"], "error": repr(err)}
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()

    summary = summarise(records)
    summary["provider"] = assistant.llm.provider
    summary["model"] = assistant.llm.model
    summary["judge_model"] = judge.model
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


METRICS = [
    "correctness",
    "citation_validity",
    "citation_accuracy",
    "gold_recall_cited",
    "gold_recall_read",
    "self_check_precision",
]


def summarise(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    summary = {"questions": len(records), "failed": len(records) - len(ok)}
    for m in METRICS:
        values = [r[m] for r in ok if r.get(m) is not None]
        summary[m] = {"mean": statistics.fmean(values), "n": len(values)} if values else None
    summary["contradictions"] = sum(r.get("contradictions", 0) for r in ok)
    return summary


def format_summary(summary: dict) -> str:
    rows = ["| Metric | Mean | n |", "|---|---|---|"]
    for m in METRICS:
        s = summary.get(m)
        rows.append(f"| {m} | {s['mean']:.3f} | {s['n']} |" if s else f"| {m} | - | 0 |")
    rows.append(
        f"\n{summary['questions']} questions, {summary['failed']} failed, "
        f"{summary['contradictions']} contradicted key points."
    )
    return "\n".join(rows)
