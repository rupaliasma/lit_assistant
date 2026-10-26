import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from litagent.llm import AnthropicLLM
from litagent.pubmed import PubMed

FIXTURES = Path(__file__).parent / "fixtures"
EMPTY_SET = "<?xml version='1.0'?><PubmedArticleSet></PubmedArticleSet>"
KNOWN = {"40189690", "10542355"}  # the papers in fixtures/efetch.xml


def ncbi_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("esearch.fcgi"):
        return httpx.Response(200, text=(FIXTURES / "esearch.json").read_text())
    if path.endswith("efetch.fcgi"):
        ids = set(request.url.params["id"].split(","))
        return httpx.Response(200, text=(FIXTURES / "efetch.xml").read_text() if ids & KNOWN else EMPTY_SET)
    if path.endswith("PMC11973158/fullTextXML"):
        return httpx.Response(200, text=(FIXTURES / "fulltext.xml").read_text())
    return httpx.Response(404)


@pytest.fixture
def pubmed():
    pm = PubMed(http=httpx.Client(transport=httpx.MockTransport(ncbi_handler)))
    pm.min_interval = 0
    return pm


# --- a stand-in for anthropic.Anthropic -------------------------------------------------


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, input, id=None):
    return SimpleNamespace(type="tool_use", name=name, input=input, id=id or f"tu_{name}")


def reply(*blocks, stop_reason=None):
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=100, output_tokens=20)
    )


def json_reply(obj):
    return reply(text_block(json.dumps(obj)))


class FakeAnthropic:
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot messages: the caller keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


@pytest.fixture
def fake_llm():
    def make(*responses, **kwargs):
        return AnthropicLLM(client=FakeAnthropic(responses), model="test-model", **kwargs)

    return make
