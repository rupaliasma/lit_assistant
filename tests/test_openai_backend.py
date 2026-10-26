import json

import httpx
import pytest

from litagent.agents import LiteratureAssistant
from litagent.llm import OpenAICompatibleLLM, make_llm, parse_json


def message(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def scripted_llm(*responses):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(requests) - 1])

    llm = OpenAICompatibleLLM(
        model="qwen-test", base_url="http://local/v1", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return llm, requests


def test_pipeline_runs_on_an_openai_compatible_server(pubmed):
    llm, requests = scripted_llm(
        message(tool_calls=[("get_abstracts", {"pmids": ["40189690"]})]),
        message(tool_calls=[("record_evidence", {"pmid": "40189690", "finding": "Error fell to 0.6 C."})]),
        message("Done."),
        message("Fourier-PD lowered the temperature error to about 0.6 degrees [PMID:40189690]."),
        # Small models often wrap JSON in prose; it should still be parsed.
        message('Sure: {"verdicts": [{"claim": 0, "pmid": "40189690", "verdict": "supported", "reason": "ok"}]}'),
    )
    result = LiteratureAssistant(llm=llm, pubmed=pubmed).ask("How accurate is Fourier-PD thermometry?")

    assert [(c.pmid, c.status) for c in result.checks] == [("40189690", "supported")]
    assert result.usage == {
        "input_tokens": 35,
        "output_tokens": 15,
        "calls": 5,
        "provider": "openai-compatible",
        "model": "qwen-test",
    }
    assert requests[0]["tools"][0]["type"] == "function" and requests[0]["messages"][0]["role"] == "system"
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "c0",
        "content": requests[1]["messages"][-1]["content"],
    }
    assert requests[4]["response_format"]["type"] == "json_schema"


def test_unusable_checker_output_is_not_fatal(pubmed):
    llm, _ = scripted_llm(
        message(tool_calls=[("get_abstracts", {"pmids": ["40189690"]})]),
        message(tool_calls=[("record_evidence", {"pmid": "40189690", "finding": "f"})]),
        message("Done."),
        message("A claim that cites the paper properly [PMID:40189690]."),
        message("I cannot do that."),
    )
    result = LiteratureAssistant(llm=llm, pubmed=pubmed, revise=False).ask("Some question about MRI?")
    assert result.checks[0].status == "unsupported" and result.checks[0].reason == "not assessed"


def test_parse_json_and_factory():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")
    assert make_llm("openai-compatible", "llama3.2:1b").model == "llama3.2:1b"
    with pytest.raises(ValueError):
        make_llm("gpt-local")


def test_json_falls_back_when_server_rejects_json_schema():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(422, json={"error": "json_schema not supported"})
        return httpx.Response(200, json=message('{"ok": true}'))

    llm = OpenAICompatibleLLM(
        model="m", base_url="http://local/v1", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert llm.json(system="s", prompt="p", schema={"type": "object"}) == {"ok": True}
    assert seen == [True, False]
