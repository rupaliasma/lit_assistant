# litagent

A multi-agent assistant that answers questions about MRI reconstruction methods from the
PubMed literature, with every statement linked to the paper that supports it and every
citation checked before the answer is returned. It runs as a FastAPI service in Docker.

```
$ litagent ask "How does GRAPPA differ from SENSE?"

SENSE reconstructs undersampled multi-coil data in image space, using the receiver coil
sensitivity maps to unfold the aliased images [PMID:10542355]. GRAPPA works in k-space
instead: missing lines are synthesised from acquired neighbouring lines of all coils, with
weights calibrated on a small fully sampled autocalibration region [PMID:12111967]. ...

[PMID:10542355] SENSE: sensitivity encoding for fast MRI. (Magn Reson Med, 1999) https://pubmed.ncbi.nlm.nih.gov/10542355/
[PMID:12111967] Generalized autocalibrating partially parallel acquisitions (GRAPPA). (Magn Reson Med, 2002) https://pubmed.ncbi.nlm.nih.gov/12111967/
```

(Illustrative output; wording varies between runs.)

## How it works

```
question
   |
   v
Researcher  --tools-->  search_pubmed / get_abstracts / get_full_text / record_evidence
   |                    (NCBI E-utilities, Europe PMC open-access full text)
   v
recorded evidence (papers read + findings)
   |
   v
Writer  -> draft answer, every factual sentence cited as [PMID:n]
   |
   v
Citation checker
   - PMID exists in PubMed?                      (deterministic)
   - paper actually read by the researcher?      (deterministic)
   - does its abstract/findings support the sentence?   (LLM, structured output)
   |
   +-- problems found -> Writer revises once -> checked again
   v
answer + sources + per-citation verdicts
```

* The **researcher** is a tool-using agent. It can only cite what it has recorded with
  `record_evidence`, and it can only record evidence from papers it has actually opened,
  which keeps the writer away from half-remembered references.
* The **writer** sees nothing but the question and the recorded evidence.
* The **checker** combines hard checks (the PMID must exist and must have been read) with
  a model judgement of support for each sentence/citation pair. Anything unsupported is
  sent back to the writer with the reason, once. The final verdicts are returned with the
  answer, so the caller can see which citations held up.

## Models

Two backends sit behind the same small interface (`litagent/llm.py`):

| Provider | What it talks to | Default model |
|---|---|---|
| `anthropic` (default) | Claude through the Anthropic SDK | `claude-opus-5-5` |
| `openai-compatible` | any OpenAI-style chat-completions server: free, local open-weight models through [Ollama](https://ollama.com), llama.cpp, vLLM or LM Studio, or a hosted open-model API | `qwen2.5:7b` |

Choose with `--provider`, `--model` and `--base-url` on the command line, or with the
`LITAGENT_PROVIDER`, `LITAGENT_MODEL`, `LITAGENT_BASE_URL` (default
`http://localhost:11434/v1`, i.e. Ollama) and `LITAGENT_LLM_API_KEY` environment
variables, which the API server also reads.

With Claude, requests opt into the API's server-side fallback, so a request the model
declines on policy grounds is retried on a fallback model instead of failing; set
`LITAGENT_FALLBACKS=0` to turn that off.

### Free, local models

```bash
# Ollama
ollama pull qwen2.5:7b
litagent ask --provider openai-compatible --model qwen2.5:7b "How does GRAPPA differ from SENSE?"

# llama.cpp via llama-cpp-python, with a GGUF file from Hugging Face
pip install "llama-cpp-python[server]"
python -m llama_cpp.server --model qwen2.5-1.5b-instruct-q4_k_m.gguf --model_alias qwen2.5-1.5b \
    --chat_format chatml-function-calling --n_ctx 16384 --port 8080
litagent ask --provider openai-compatible --base-url http://localhost:8080/v1 \
    --model qwen2.5-1.5b --max-turns 6 "How does GRAPPA differ from SENSE?"
```

The model must support tool (function) calling. Servers that reject the `json_schema`
response format (llama-cpp-python, for one) are handled by retrying with a plain JSON
request.

What to expect from small models, from smoke tests on a 4-core CPU with Qwen2.5 GGUF
models (Q4_K_M): with 0.5B the pipeline runs end to end but the model cannot produce
tool calls, so no papers are read; 1.5B calls the PubMed tools but tends to repeat the
same search, and when it invented a citation the checker caught it (the PMID existed but
had never been read). They are fine for checking that everything works; for useful
answers use a 7B-class model or larger (Qwen2.5 7B/14B, Llama 3.1 8B) or Claude, and a
capable model as evaluation judge.

## Installation

```bash
pip install -e "lit-assistant[dev]"
export ANTHROPIC_API_KEY=...           # only for the anthropic provider
export NCBI_EMAIL=you@example.org      # optional, NCBI asks for it
export NCBI_API_KEY=...                # optional, 10 instead of 3 requests per second
```

## Usage

Command line:

```bash
litagent ask "What limits the acceleration achievable with SENSE?"
litagent ask --json "How is k-space interpolated in RAKI?" > answer.json
litagent serve --port 8000
```

HTTP API (docs at `http://localhost:8000/docs` once running):

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How does MoDL combine deep learning with model-based reconstruction?"}'
```

The response holds `answer`, `sources` (PMID, title, journal, year, authors, DOI and
PubMed URL), `checks` (one verdict per sentence/citation pair), `citation_precision`,
`uncited` (sentences that make a claim without a citation), `papers_read` and token
`usage`. If `LITAGENT_API_KEY` is set, requests must carry it in an `X-API-Key` header.
Most of the time per question is spent in the research loop, so expect answers to take
minutes rather than seconds.

Docker:

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY, or the LITAGENT_PROVIDER settings
docker compose up --build     # or: docker run --env-file .env -p 8000:8000 ghcr.io/soumickmj/litagent
```

To use Ollama running on the host from inside the container, set
`LITAGENT_BASE_URL=http://host.docker.internal:11434/v1` (on Linux also add
`--add-host=host.docker.internal:host-gateway`).

## Evaluation

`data/questions.jsonl` holds 22 hand-written questions covering classical and
learning-based reconstruction: compressed sensing, SENSE, GRAPPA, ESPIRiT, variational
networks, MoDL, AUTOMAP, cascades, learned primal-dual, Fourier-PD, KIKI-net, k-space
learning, RAKI, self-supervised training, fastMRI, robustness, generalisation, diffusion
models and complex-valued networks. Each question lists the key points a good answer
should make and the PubMed IDs of the papers that establish them. The key points were
checked against the abstracts of those papers, and `tests/test_live.py` confirms that
every gold PMID exists.

```bash
scripts/run_eval.sh                # all 22 questions, results in results/<timestamp>/
LIMIT=3 scripts/run_eval.sh        # quick check
PROVIDER=openai-compatible MODEL=qwen2.5:7b LIMIT=1 scripts/run_eval.sh   # local model
```

Metrics, per question and averaged:

| Metric | What it measures |
|---|---|
| correctness | share of key points covered (partial counts half), graded by a separate judge prompt that sees only the answer and the key points |
| citation_validity | share of cited PMIDs that exist in PubMed |
| citation_accuracy | share of sentence/citation pairs whose cited abstract supports the sentence, judged by an independent reviewer prompt; citations of non-existent papers count as unsupported |
| gold_recall_cited / gold_recall_read | share of the curated key papers that were cited / at least read |
| self_check_precision | the pipeline's own citation-checker score, to compare with the independent one |

Contradicted key points are counted separately. `answers.jsonl` keeps every answer with
its grades, so individual cases can be inspected.

LLM judges are not ground truth. The correctness and citation-accuracy graders are worth
spot-checking by hand on a sample of `answers.jsonl` before quoting numbers, and the
judge model can be set apart from the answering model with `--judge-model`.

### Results

Not yet run in full. A one-question smoke run with Qwen2.5-1.5B as both assistant and
judge completed, but the judge's JSON did not match the grading schema, so the question
was reported as failed rather than scored (by design: malformed judgements are errors,
not zeros). After `scripts/run_eval.sh` with a capable model, paste the table printed at
the end (also in `results/<run>/summary.json`) here, together with the provider, model,
judge model and date.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest                                   # offline, a second or so
LITAGENT_LIVE_TESTS=1 pytest -m live     # talks to NCBI and Europe PMC
```

The offline tests mock NCBI with recorded responses (`tests/fixtures/`) and replace both
model backends with scripted fakes, so the whole pipeline (research loop, tool errors,
revision round, citation checks), the API and the evaluation harness are exercised
without network access or API costs.

CI (`.github/workflows/litagent.yml` at the repository root) lints and tests every push
and pull request and builds the Docker image; a `litagent-v*` tag publishes it to GHCR.

## Licence

MIT. Paper metadata comes from NCBI PubMed and full text from Europe PMC; please respect
their usage policies (https://www.ncbi.nlm.nih.gov/home/about/policies/ and
https://europepmc.org/Copyright).
