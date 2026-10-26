"""Hits the real NCBI and Europe PMC services. Run with LITAGENT_LIVE_TESTS=1 pytest -m live."""

import json
import os

import pytest
from test_evaluate import QUESTIONS

from litagent.pubmed import PubMed

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("LITAGENT_LIVE_TESTS") != "1", reason="set LITAGENT_LIVE_TESTS=1"),
]


@pytest.fixture(scope="module")
def pubmed():
    return PubMed()


def test_search_finds_a_known_paper(pubmed):
    assert "17969013" in pubmed.search("Lustig sparse MRI compressed sensing rapid MR imaging", 5)


def test_full_text_of_open_access_paper(pubmed):
    text = pubmed.full_text("40189690")
    assert text and "thermometry" in text.lower()


def test_every_gold_pmid_exists(pubmed):
    gold = sorted({p for line in QUESTIONS.read_text().splitlines() for p in json.loads(line)["gold_pmids"]})
    found = {p.pmid for p in pubmed.fetch(gold)}
    assert found == set(gold)
