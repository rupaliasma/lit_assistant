"""Tools exposed to the research agent, plus the evidence they collect."""

from dataclasses import dataclass, field

from litagent.pubmed import Paper, PubMed

FULL_TEXT_PAGE = 12_000


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOLS = [
    {
        "name": "search_pubmed",
        "description": (
            "Search PubMed and return matching papers (PMID, year, first author, title, journal), most relevant "
            "first. Accepts PubMed query syntax, e.g. 'compressed sensing[Title] AND MRI'. Run several focused "
            "queries rather than one broad one."
        ),
        "input_schema": _schema(
            {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1 to 20, default 10"},
            },
            ["query", "max_results"],
        ),
        "strict": True,
    },
    {
        "name": "get_abstracts",
        "description": "Fetch title, journal, year, DOI and abstract for up to 10 PMIDs.",
        "input_schema": _schema({"pmids": {"type": "array", "items": {"type": "string"}}}, ["pmids"]),
        "strict": True,
    },
    {
        "name": "get_full_text",
        "description": (
            "Fetch the body text of an open-access paper from Europe PMC, in pages of about 12,000 characters. "
            "Use only when the abstract lacks a detail you need. Returns a notice if no open-access text exists."
        ),
        "input_schema": _schema(
            {"pmid": {"type": "string"}, "offset": {"type": "integer", "description": "character offset, start at 0"}},
            ["pmid", "offset"],
        ),
        "strict": True,
    },
    {
        "name": "record_evidence",
        "description": (
            "Record a finding from a paper you have read with get_abstracts or get_full_text. State the finding "
            "as the paper supports it, in one or two sentences. Only recorded papers can be cited in the answer."
        ),
        "input_schema": _schema({"pmid": {"type": "string"}, "finding": {"type": "string"}}, ["pmid", "finding"]),
        "strict": True,
    },
]


class ToolError(Exception):
    pass


@dataclass
class ToolBox:
    pubmed: PubMed
    read: dict[str, Paper] = field(default_factory=dict)  # papers whose abstract or text the agent has seen
    findings: dict[str, list[str]] = field(default_factory=dict)
    searches: list[str] = field(default_factory=list)

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        try:
            return handler(**args)
        except (TypeError, ValueError) as err:
            raise ToolError(str(err)) from err

    def _tool_search_pubmed(self, query: str, max_results: int = 10) -> str:
        self.searches.append(query)
        pmids = self.pubmed.search(query, max(1, min(int(max_results), 20)))
        if not pmids:
            return "No results. Try broader terms or synonyms."
        lines = [f"{p.pmid} | {p.year} | {p.short_ref()} | {p.title}" for p in self.pubmed.fetch(pmids)]
        return "\n".join(lines)

    def _tool_get_abstracts(self, pmids: list[str]) -> str:
        if len(pmids) > 10:
            raise ValueError("at most 10 PMIDs per call")
        papers = self.pubmed.fetch(pmids)
        found = {p.pmid for p in papers}
        out = []
        for p in papers:
            self.read[p.pmid] = p
            out.append(
                f"PMID {p.pmid} | {p.short_ref()} | DOI {p.doi or '-'} | open access: {'yes' if p.pmcid else 'no'}\n"
                f"Title: {p.title}\nAbstract: {p.abstract or '(no abstract)'}"
            )
        out += [f"PMID {pmid}: not found in PubMed" for pmid in pmids if pmid.strip() not in found]
        return "\n\n".join(out)

    def _tool_get_full_text(self, pmid: str, offset: int = 0) -> str:
        text = self.pubmed.full_text(pmid)
        if text is None:
            return f"No open-access full text for PMID {pmid}; rely on the abstract."
        self.read.setdefault(pmid, self.pubmed.fetch([pmid])[0])
        offset = max(0, int(offset))
        end = min(len(text), offset + FULL_TEXT_PAGE)
        more = f" Call again with offset={end} for the next page." if end < len(text) else ""
        return f"[characters {offset}-{end} of {len(text)}.{more}]\n{text[offset:end]}"

    def _tool_record_evidence(self, pmid: str, finding: str) -> str:
        pmid = pmid.strip()
        if pmid not in self.read:
            raise ValueError(f"read PMID {pmid} with get_abstracts before recording evidence from it")
        self.findings.setdefault(pmid, []).append(finding.strip())
        return f"Recorded ({sum(map(len, self.findings.values()))} findings from {len(self.findings)} papers so far)."

    def evidence(self) -> list[dict]:
        """Recorded papers with their findings, in the order they were first recorded."""
        return [{"paper": self.read[pmid], "findings": notes} for pmid, notes in self.findings.items()]

    def evidence_block(self) -> str:
        blocks = []
        for item in self.evidence():
            p = item["paper"]
            findings = "\n".join(f"- {f}" for f in item["findings"])
            blocks.append(
                f"[PMID:{p.pmid}] {p.short_ref()}. {p.title}\nAbstract: {p.abstract}\nRecorded findings:\n{findings}"
            )
        return "\n\n".join(blocks)
