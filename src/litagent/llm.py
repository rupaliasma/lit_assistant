"""Language-model backends behind one small interface.

The agents need three things from a model: a tool-use conversation, a plain
completion and a JSON-constrained completion. Two backends provide them:

* ``anthropic``: Claude through the Anthropic SDK (default).
* ``openai-compatible``: any server speaking the OpenAI chat-completions API,
  which covers free, local open-weight models via Ollama, llama.cpp
  (``llama-server``), vLLM or LM Studio, and hosted open-model APIs.

Pick one with ``make_llm`` or the ``LITAGENT_PROVIDER``, ``LITAGENT_MODEL``,
``LITAGENT_BASE_URL`` and ``LITAGENT_LLM_API_KEY`` environment variables.
"""

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai-compatible")
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai-compatible": "qwen2.5:7b"}
DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama
# Server-side fallback (Anthropic): if the model declines a request on policy grounds,
# the API re-runs it on a fallback model chosen by refusal category. Disable with
# LITAGENT_FALLBACKS=0 (e.g. behind a proxy or on a platform that lacks it).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens, output_tokens) -> None:
        self.calls += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict | None  # None when the model produced arguments that are not valid JSON


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_json(text: str) -> dict:
    """JSON object from a model reply, tolerating code fences or prose around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in model output: {text[:200]!r}") from None
        return json.loads(match.group(0))


# --- Anthropic ----------------------------------------------------------------------------


class AnthropicConversation:
    def __init__(self, llm, system, tools, effort):
        self.llm, self.system, self.tools, self.effort = llm, system, tools, effort
        self.messages = []

    def _call(self) -> Reply:
        response = self.llm.create(system=self.system, messages=self.messages, tools=self.tools, effort=self.effort)
        # The full content goes back unchanged, thinking blocks included.
        self.messages.append({"role": "assistant", "content": response.content})
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        return Reply(_text_of(response), calls)

    def send(self, content: str) -> Reply:
        self.messages.append({"role": "user", "content": content})
        return self._call()

    def send_tool_results(self, results: list[tuple[str, str, bool]]) -> Reply:
        blocks = [{"type": "tool_result", "tool_use_id": i, "content": c, "is_error": e} for i, c, e in results]
        self.messages.append({"role": "user", "content": blocks})
        return self._call()


def _text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, client=None, model: str | None = None, fallbacks: bool | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.fallbacks = os.environ.get("LITAGENT_FALLBACKS", "1") != "0" if fallbacks is None else fallbacks
        self.usage = Usage()

    def fork(self) -> "AnthropicLLM":
        """Same client and settings, fresh usage counter."""
        return AnthropicLLM(self.client, self.model, self.fallbacks)

    def create(self, *, system, messages, tools=None, effort="medium", max_tokens=16000, schema=None):
        output_config = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs = dict(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages, output_config=output_config
        )
        if tools:
            kwargs["tools"] = tools
        if self.fallbacks:
            kwargs |= dict(betas=[FALLBACK_BETA], fallbacks="default")

        response = self.client.beta.messages.create(**kwargs)
        self.usage.add(getattr(response.usage, "input_tokens", 0), getattr(response.usage, "output_tokens", 0))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RefusalError(f"model declined the request: {getattr(details, 'category', None)}")
        if response.stop_reason == "max_tokens":
            log.warning("response hit max_tokens=%d and may be incomplete", max_tokens)
        return response

    def conversation(self, system: str, tools: list, effort: str = "medium") -> AnthropicConversation:
        return AnthropicConversation(self, system, tools, effort)

    def complete(self, *, system: str, prompt: str, effort: str = "medium") -> str:
        return _text_of(self.create(system=system, messages=[{"role": "user", "content": prompt}], effort=effort))

    def json(self, *, system: str, prompt: str, schema: dict, effort: str = "low") -> dict:
        response = self.create(
            system=system, messages=[{"role": "user", "content": prompt}], schema=schema, effort=effort
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError("structured response was cut off at max_tokens")
        return json.loads(_text_of(response))


# --- OpenAI-compatible (Ollama, llama.cpp, vLLM, LM Studio, ...) ---------------------------


def _openai_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]},
    }


class OpenAIConversation:
    def __init__(self, llm, system, tools):
        self.llm = llm
        self.tools = [_openai_tool(t) for t in tools]
        self.messages = [{"role": "system", "content": system}]

    def _call(self) -> Reply:
        message = self.llm.chat(self.messages, tools=self.tools)
        calls = []
        for call in message.get("tool_calls") or []:
            call_id = call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            call["id"] = call_id  # some local servers omit IDs; the next turn needs them
            raw = call["function"].get("arguments") or "{}"
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw)
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(call_id, call["function"]["name"], args))
        self.messages.append(
            {"role": "assistant", "content": message.get("content") or "", "tool_calls": message.get("tool_calls")}
            if calls
            else {"role": "assistant", "content": message.get("content") or ""}
        )
        return Reply((message.get("content") or "").strip(), calls)

    def send(self, content: str) -> Reply:
        self.messages.append({"role": "user", "content": content})
        return self._call()

    def send_tool_results(self, results: list[tuple[str, str, bool]]) -> Reply:
        for call_id, content, _ in results:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        return self._call()


class OpenAICompatibleLLM:
    provider = "openai-compatible"

    def __init__(
        self, model=None, base_url=None, api_key=None, http: httpx.Client | None = None, max_tokens: int = 4096
    ):
        self.model = model or DEFAULT_MODELS["openai-compatible"]
        self.base_url = (base_url or os.environ.get("LITAGENT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LITAGENT_LLM_API_KEY")
        # Local models on a CPU can take minutes per call.
        self.http = http or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))
        self.max_tokens = max_tokens
        self.usage = Usage()

    def fork(self) -> "OpenAICompatibleLLM":
        return OpenAICompatibleLLM(self.model, self.base_url, self.api_key, self.http, self.max_tokens)

    def chat(self, messages: list, tools: list | None = None, response_format: dict | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if tools:
            body |= {"tools": tools, "tool_choice": "auto"}
        if response_format:
            body["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self.http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        self.usage.add(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            log.warning("response hit max_tokens=%d and may be incomplete", self.max_tokens)
        return choice["message"]

    def conversation(self, system: str, tools: list, effort: str = "medium") -> OpenAIConversation:
        return OpenAIConversation(self, system, tools)  # effort has no equivalent here

    def complete(self, *, system: str, prompt: str, effort: str = "medium") -> str:
        message = self.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return (message.get("content") or "").strip()

    def json(self, *, system: str, prompt: str, schema: dict, effort: str = "low") -> dict:
        fmt = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}
        prompt += "\n\nReply with a single JSON object matching this schema:\n" + json.dumps(schema)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        try:
            message = self.chat(messages, response_format=fmt)
        except httpx.HTTPStatusError as err:
            # Not every server takes json_schema (llama-cpp-python answers 500, others 400/422); the prompt
            # itself asks for JSON, so retry once without it. A real server fault fails again here.
            log.info("request with json_schema failed (%s), retrying without it", err.response.status_code)
            message = self.chat(messages)
        return parse_json(message.get("content") or "")


def make_llm(provider: str | None = None, model: str | None = None, base_url: str | None = None):
    provider = provider or os.environ.get("LITAGENT_PROVIDER", "anthropic")
    model = model or os.environ.get("LITAGENT_MODEL")
    if provider == "anthropic":
        return AnthropicLLM(model=model)
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(model=model, base_url=base_url)
    raise ValueError(f"unknown provider {provider!r}, choose from {PROVIDERS}")
