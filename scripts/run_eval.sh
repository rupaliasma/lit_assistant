#!/usr/bin/env bash
# Run the evaluation set and keep the results in a dated folder.
#
#   ANTHROPIC_API_KEY=... scripts/run_eval.sh                          # all questions, Claude
#   LIMIT=3 scripts/run_eval.sh                                        # quick check
#   PROVIDER=openai-compatible MODEL=qwen2.5:7b scripts/run_eval.sh    # local model via Ollama
set -euo pipefail

PROVIDER=${PROVIDER:-${LITAGENT_PROVIDER:-anthropic}}
[[ $PROVIDER == anthropic ]] && : "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
OUT=${OUT:-results/$(date +%Y%m%d-%H%M%S)}
args=(--questions "${QUESTIONS:-data/questions.jsonl}" --out "$OUT" --provider "$PROVIDER")
[[ -n ${BASE_URL:-} ]] && args+=(--base-url "$BASE_URL")
[[ -n ${LIMIT:-} ]] && args+=(--limit "$LIMIT")
[[ -n ${MODEL:-} ]] && args+=(--model "$MODEL")
[[ -n ${JUDGE_MODEL:-} ]] && args+=(--judge-model "$JUDGE_MODEL")

mkdir -p "$OUT"
litagent -v eval "${args[@]}" 2>&1 | tee "$OUT/eval.log"
