"""Parsing of inline ``[PMID:123]`` citations into claim/citation pairs."""

import re
from dataclasses import dataclass

BRACKET_RE = re.compile(r"\[(?:PMID|PubMed)\s*:?[^\]]*\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
LEADING_BRACKETS_RE = re.compile(r"^((?:\s*\[(?:PMID|PubMed)[^\]]*\])+)[\s.]*(.*)$", re.IGNORECASE)


@dataclass
class Claim:
    text: str
    pmids: list[str]


def cited_pmids(text: str) -> list[str]:
    """All PMIDs cited in ``text``, in order of first appearance."""
    found = []
    for bracket in BRACKET_RE.findall(text):
        found += NUMBER_RE.findall(bracket)
    return list(dict.fromkeys(found))


def split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.split("\n"):
        for chunk in SENTENCE_END_RE.split(paragraph.strip()):
            if not chunk:
                continue
            # Citations placed after the full stop belong to the previous sentence.
            lead = LEADING_BRACKETS_RE.match(chunk)
            if sentences and lead:
                sentences[-1] += " " + lead.group(1).strip()
                chunk = lead.group(2)
            if chunk:
                sentences.append(chunk)
    return sentences


def is_statement(sentence: str) -> bool:
    """Rough filter for sentences that make a claim (not headings, list labels or very short fragments)."""
    stripped = BRACKET_RE.sub("", sentence).strip(" #*-:")
    return len(stripped.split()) >= 5 and not stripped.endswith("?")


def extract_claims(text: str) -> list[Claim]:
    return [Claim(s, cited_pmids(s)) for s in split_sentences(text) if cited_pmids(s)]


def uncited_statements(text: str) -> list[str]:
    return [s for s in split_sentences(text) if is_statement(s) and not cited_pmids(s)]
