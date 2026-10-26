import pytest
from fastapi.testclient import TestClient

from litagent.agents import Answer, CitationCheck
from litagent.api import create_app
from litagent.llm import RefusalError


class StubAssistant:
    def __init__(self, error=None):
        self.error = error
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return Answer(
            question=question,
            answer="GRAPPA interpolates missing k-space lines [PMID:12111967].",
            sources=[
                {
                    "pmid": "12111967",
                    "title": "GRAPPA",
                    "journal": "Magn Reson Med",
                    "year": "2002",
                    "authors": ["Griswold MA"],
                    "doi": "10.1002/mrm.10171",
                    "abstract": "...",
                    "pmcid": None,
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12111967/",
                }
            ],
            checks=[
                CitationCheck(
                    "GRAPPA interpolates missing k-space lines [PMID:12111967].",
                    "12111967",
                    "supported",
                    "stated in abstract",
                )
            ],
            uncited=[],
            research_notes="",
            papers_read=["12111967"],
        )


def test_health():
    assert TestClient(create_app(StubAssistant())).get("/health").json()["status"] == "ok"


def test_ask_returns_answer_with_sources():
    stub = StubAssistant()
    resp = TestClient(create_app(stub)).post("/ask", json={"question": "  How does GRAPPA work in practice?  "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"][0]["url"].endswith("/12111967/")
    assert body["citation_precision"] == 1.0
    assert stub.questions == ["How does GRAPPA work in practice?"]


@pytest.mark.parametrize("payload", [{}, {"question": "short"}, {"question": "x" * 1001}])
def test_rejects_bad_questions(payload):
    assert TestClient(create_app(StubAssistant())).post("/ask", json=payload).status_code == 422


def test_refusal_is_reported():
    client = TestClient(create_app(StubAssistant(error=RefusalError("declined"))))
    resp = client.post("/ask", json={"question": "How does GRAPPA work in practice?"})
    assert resp.status_code == 422 and "declined" in resp.json()["detail"]


def test_api_key(monkeypatch):
    monkeypatch.setenv("LITAGENT_API_KEY", "s3cret")
    client = TestClient(create_app(StubAssistant()))
    q = {"question": "How does GRAPPA work in practice?"}
    assert client.post("/ask", json=q).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200
