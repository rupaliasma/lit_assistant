"""The three agents and the pipeline that runs them.

1. The researcher searches PubMed, reads abstracts (and open-access full text
   when needed) and records findings, in a tool-use loop.
2. The writer drafts an answer from the recorded evidence only, citing every
   statement as [PMID:n].
3. The citation checker verifies each cited PMID (it must exist and have been
   read by the researcher) and asks the model whether the cited abstract and
   findings actually support the sentence. Unsupported citations go back to
   the writer for one revision.
"""

import logging
from dataclasses import asdict, dataclass, field

from litagent.citations import extract_claims, uncited_statements
from litagent.llm import make_llm
from litagent.pubmed import PubMed
from litagent.tools import TOOLS, ToolBox, ToolError

log = logging.getLogger(__name__)

RESEARCHER_SYSTEM = """\
You are the literature researcher in a team that answers questions about MRI reconstruction \
methods. Your job is to find and read the papers that answer the question, not to write the answer.

- Search PubMed with several focused queries: method names, authors you know of, synonyms. \
Prefer the original papers that introduced a method, plus strong evaluations or reviews.
- Read abstracts with get_abstracts before relying on a paper. Use get_full_text only when an \
abstract lacks a detail the question depends on.
- For every paper you will rely on, call record_evidence with what that paper actually reports, \
in your own words, specific to the question. Never record anything you have not read in the \
retrieved text, even if you believe it to be true.
- Four to ten well-chosen papers are usually enough. Stop when the question is covered.

When you are done, reply with two or three sentences on what the evidence covers and any gaps."""

WRITER_SYSTEM = """\
You write answers to questions about MRI reconstruction for researchers, using only the evidence \
provided. Rules:

- Every sentence that states a fact must end with its supporting citation(s) in the form \
[PMID:12345678], placed before the full stop. Cite several as [PMID:1; PMID:2].
- Cite only PMIDs that appear in the evidence, and only for what that paper's abstract or recorded \
findings support. Do not add facts from memory.
- If the evidence does not answer part of the question, say so plainly instead of guessing.
- Be precise and concise: usually 150 to 400 words of plain prose, British English, no headings, \
no reference list (sources are listed separately)."""

CHECKER_SYSTEM = """\
You check citations in scientific answers. For each sentence and each PMID cited in it, decide \
whether the cited paper, as described by its abstract and the recorded findings, supports the \
sentence: "supported" if it states or directly implies the claim, "partial" if it supports only \
part of it or something weaker, "unsupported" if it does not support it or contradicts it. Judge \
only against the text given, not your own knowledge. Keep each reason to one sentence."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "integer"},
                    "pmid": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "pmid", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


@dataclass
class CitationCheck:
    claim: str
    pmid: str
    status: str  # supported | partial | unsupported | not_in_evidence | not_found
    reason: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict]
    checks: list[CitationCheck]
    uncited: list[str]
    research_notes: str
    papers_read: list[str]
    revised: bool = False
    usage: dict = field(default_factory=dict)

    @property
    def citation_precision(self) -> float | None:
        """Share of claim/citation pairs judged supported (partial counts half)."""
        if not self.checks:
            return None
        score = {"supported": 1.0, "partial": 0.5}
        return sum(score.get(c.status, 0.0) for c in self.checks) / len(self.checks)

    def to_dict(self) -> dict:
        return asdict(self) | {"citation_precision": self.citation_precision}


def run_researcher(question: str, toolbox: ToolBox, llm, max_turns: int = 15) -> str:
    conversation = llm.conversation(RESEARCHER_SYSTEM, TOOLS, effort="medium")
    reply = conversation.send(f"Question: {question}")
    for turn in range(max_turns):
        if not reply.tool_calls:
            return reply.text
        results = []
        for call in reply.tool_calls:
            try:
                if call.args is None:
                    raise ToolError("arguments were not valid JSON")
                content, is_error = toolbox.run(call.name, call.args), False
            except ToolError as err:
                content, is_error = f"Error: {err}", True
            log.info("researcher turn %d: %s(%s) -> %s", turn, call.name, call.args, content[:80].replace("\n", " "))
            results.append((call.id, content, is_error))
        if turn < max_turns - 1:
            reply = conversation.send_tool_results(results)
    log.warning("researcher stopped after %d turns", max_turns)
    return f"(stopped after {max_turns} turns; evidence may be incomplete)"


def write_answer(question: str, evidence: str, llm, feedback: str | None = None, draft: str | None = None) -> str:
    prompt = f"Question: {question}\n\nEvidence:\n{evidence or '(no evidence was found)'}"
    if feedback:
        prompt += (
            f"\n\nYour previous draft:\n{draft}\n\nA citation check found these problems:\n{feedback}\n\n"
            "Rewrite the answer so that every citation is supported: fix the sentence, cite a paper that does "
            "support it, or drop the claim."
        )
    return llm.complete(system=WRITER_SYSTEM, prompt=prompt, effort="medium")


def check_citations(answer: str, toolbox: ToolBox, pubmed: PubMed, llm) -> list[CitationCheck]:
    claims = extract_claims(answer)
    checks: dict[tuple[int, str], CitationCheck] = {}
    to_judge = []
    for i, claim in enumerate(claims):
        for pmid in claim.pmids:
            if pmid in toolbox.findings:
                to_judge.append((i, pmid))
            elif pubmed.exists(pmid):
                checks[i, pmid] = CitationCheck(
                    claim.text, pmid, "not_in_evidence", "paper was not read by the researcher"
                )
            else:
                checks[i, pmid] = CitationCheck(claim.text, pmid, "not_found", "PMID does not exist in PubMed")

    if to_judge:
        papers = {pmid: toolbox.read[pmid] for _, pmid in to_judge}
        sources = "\n\n".join(
            f"PMID {p.pmid}: {p.title}\nAbstract: {p.abstract}\nRecorded findings: "
            + " ".join(toolbox.findings.get(p.pmid, []))
            for p in papers.values()
        )
        pairs = "\n".join(f"claim {i}, PMID {pmid}: {claims[i].text}" for i, pmid in to_judge)
        try:
            result = llm.json(
                system=CHECKER_SYSTEM, prompt=f"Sources:\n{sources}\n\nCheck these:\n{pairs}", schema=VERDICT_SCHEMA
            )
            verdicts = result.get("verdicts", []) if isinstance(result, dict) else []
        except ValueError as err:  # small local models sometimes return malformed JSON
            log.warning("citation check returned unusable output: %s", err)
            verdicts = []
        wanted = set(to_judge)
        for v in verdicts:
            try:
                key = (int(v["claim"]), str(v["pmid"]))
                status = v["verdict"] if v["verdict"] in ("supported", "partial", "unsupported") else "unsupported"
            except (KeyError, TypeError, ValueError):
                continue
            if key in wanted:
                checks[key] = CitationCheck(claims[key[0]].text, key[1], status, str(v.get("reason", "")))
        for i, pmid in to_judge:  # the judge skipped a pair: treat as unchecked, not as supported
            checks.setdefault((i, pmid), CitationCheck(claims[i].text, pmid, "unsupported", "not assessed"))

    return [checks[k] for k in sorted(checks)]


def _feedback(checks: list[CitationCheck]) -> str:
    bad = [c for c in checks if c.status not in ("supported", "partial")]
    return "\n".join(f'- "{c.claim}" cites PMID {c.pmid}: {c.status} ({c.reason})' for c in bad)


class LiteratureAssistant:
    def __init__(self, llm=None, pubmed: PubMed | None = None, max_research_turns: int = 15, revise: bool = True):
        self.llm = llm or make_llm()
        self.pubmed = pubmed or PubMed()
        self.max_research_turns = max_research_turns
        self.revise = revise

    def ask(self, question: str) -> Answer:
        llm = self.llm.fork()  # per-question token accounting, safe under concurrent requests
        toolbox = ToolBox(self.pubmed)
        notes = run_researcher(question, toolbox, llm, self.max_research_turns)
        evidence = toolbox.evidence_block()

        answer = write_answer(question, evidence, llm)
        checks = check_citations(answer, toolbox, self.pubmed, llm)
        revised = False
        feedback = _feedback(checks)
        if self.revise and feedback:
            log.info("revising answer:\n%s", feedback)
            answer = write_answer(question, evidence, llm, feedback=feedback, draft=answer)
            checks = check_citations(answer, toolbox, self.pubmed, llm)
            revised = True

        cited = {c.pmid for c in checks}
        sources = [p.to_dict() for pmid, p in toolbox.read.items() if pmid in cited]
        sources += [p.to_dict() for p in self.pubmed.fetch(sorted(cited - set(toolbox.read)))]
        return Answer(
            question=question,
            answer=answer,
            sources=sources,
            checks=checks,
            uncited=uncited_statements(answer),
            research_notes=notes,
            papers_read=list(toolbox.read),
            revised=revised,
            usage=asdict(llm.usage) | {"provider": llm.provider, "model": llm.model},
        )
