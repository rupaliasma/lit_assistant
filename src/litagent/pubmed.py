"""Small clients for NCBI E-utilities (PubMed) and Europe PMC full text.

NCBI asks for at most 3 requests per second without an API key and 10 with
one; requests are spaced accordingly. Set ``NCBI_API_KEY`` and ``NCBI_EMAIL``
in the environment to identify yourself, as their usage policy requests.
"""

import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import httpx

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str = ""
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    pmcid: str | None = None

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def short_ref(self) -> str:
        first = self.authors[0] if self.authors else "Anonymous"
        etal = " et al." if len(self.authors) > 1 else ""
        return f"{first}{etal} {self.journal} {self.year}".strip()

    def to_dict(self) -> dict:
        return asdict(self) | {"url": self.url}


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_efetch(xml: str) -> list[Paper]:
    papers = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        a = cit.find("Article")
        parts = []
        for ab in a.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
        year = (
            a.findtext("Journal/JournalIssue/PubDate/Year")
            or (a.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "")[:4]
        )
        authors = []
        for au in a.findall("AuthorList/Author"):
            last, initials = au.findtext("LastName"), au.findtext("Initials")
            if last:
                authors.append(f"{last} {initials}".strip() if initials else last)
            elif au.findtext("CollectiveName"):
                authors.append(au.findtext("CollectiveName"))
        # Only the article's own IDs; the reference list carries IDs of cited papers too.
        ids = {i.get("IdType"): i.text for i in art.findall("PubmedData/ArticleIdList/ArticleId")}
        papers.append(
            Paper(
                pmid=cit.findtext("PMID"),
                title=_text(a.find("ArticleTitle")),
                abstract="\n".join(parts),
                journal=a.findtext("Journal/ISOAbbreviation") or a.findtext("Journal/Title") or "",
                year=year,
                authors=authors,
                doi=ids.get("doi"),
                pmcid=ids.get("pmc"),
            )
        )
    return papers


def parse_jats_body(xml: str) -> str:
    """Section titles and paragraphs of a JATS article body, as plain text."""
    body = ET.fromstring(xml).find(".//body")
    if body is None:
        return ""
    lines = []
    for el in body.iter():
        if el.tag == "title":
            lines.append(f"\n## {_text(el)}")
        elif el.tag == "p":
            lines.append(_text(el))
    return "\n".join(lines).strip()


class PubMed:
    def __init__(self, http: httpx.Client | None = None, api_key: str | None = None, email: str | None = None):
        self.http = http or httpx.Client(timeout=30.0, headers={"User-Agent": "litagent/0.1"})
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL")
        self.min_interval = 0.11 if self.api_key else 0.34
        self._last = 0.0
        self._lock = threading.Lock()
        self._cache: dict[str, Paper] = {}

    def _get(self, url: str, params: dict | None = None, ncbi: bool = True) -> httpx.Response:
        params = dict(params or {})
        if ncbi:
            params["tool"] = "litagent"
            if self.api_key:
                params["api_key"] = self.api_key
            if self.email:
                params["email"] = self.email
        for attempt in range(3):
            if ncbi:
                with self._lock:
                    wait = self.min_interval - (time.monotonic() - self._last)
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.monotonic()
            resp = self.http.get(url, params=params)
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                resp.raise_for_status()
                return resp
            log.warning("%s returned %s, retrying", url, resp.status_code)
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def search(self, query: str, max_results: int = 10) -> list[str]:
        resp = self._get(
            f"{EUTILS}/esearch.fcgi",
            {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"},
        )
        return resp.json()["esearchresult"]["idlist"]

    def fetch(self, pmids: list[str]) -> list[Paper]:
        """Papers for the given PMIDs, in the same order; unknown PMIDs are left out."""
        pmids = [p.strip() for p in pmids]
        bad = [p for p in pmids if not PMID_RE.match(p)]
        if bad:
            raise ValueError(f"not a PMID: {', '.join(bad)}")
        missing = [p for p in dict.fromkeys(pmids) if p not in self._cache]
        if missing:
            resp = self._get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(missing), "retmode": "xml"})
            for paper in parse_efetch(resp.text):
                self._cache[paper.pmid] = paper
        return [self._cache[p] for p in pmids if p in self._cache]

    def exists(self, pmid: str) -> bool:
        return PMID_RE.match(pmid) is not None and bool(self.fetch([pmid]))

    def full_text(self, pmid: str) -> str | None:
        """Body text from Europe PMC for open-access papers, otherwise ``None``."""
        papers = self.fetch([pmid])
        if not papers or not papers[0].pmcid:
            return None
        try:
            resp = self._get(f"{EUROPEPMC}/{papers[0].pmcid}/fullTextXML", ncbi=False)
        except httpx.HTTPStatusError as err:
            if err.response.status_code == 404:
                return None
            raise
        return parse_jats_body(resp.text) or None
