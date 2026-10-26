import argparse
import json
import logging

from litagent.llm import PROVIDERS


def _llm_args(p):
    p.add_argument("--provider", choices=PROVIDERS, help="default: $LITAGENT_PROVIDER or anthropic")
    p.add_argument("--model", help="default: $LITAGENT_MODEL, else claude-opus-5-5 / qwen2.5:7b")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint, default $LITAGENT_BASE_URL or Ollama on localhost")
    p.add_argument("--max-turns", type=int, default=15, help="tool-use turns for the researcher")


def build_parser():
    parser = argparse.ArgumentParser(prog="litagent", description="Cited answers on MRI reconstruction from PubMed")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full result as JSON")
    _llm_args(a)

    e = sub.add_parser("eval", help="run the evaluation question set")
    e.add_argument("--questions", default="data/questions.jsonl")
    e.add_argument("--judge-model", help="model for the graders (same provider); default: the answering model")
    e.add_argument("--limit", type=int, help="only the first N questions")
    e.add_argument("--out", default="results")
    _llm_args(e)

    s = sub.add_parser("serve", help="run the HTTP API (configured through LITAGENT_* environment variables)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "serve":
        import uvicorn

        uvicorn.run("litagent.api:app", host=args.host, port=args.port)
        return

    from litagent.agents import LiteratureAssistant
    from litagent.llm import make_llm

    llm = make_llm(args.provider, args.model, args.base_url)
    assistant = LiteratureAssistant(llm=llm, max_research_turns=args.max_turns)
    if args.command == "ask":
        result = assistant.ask(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
            return
        print(result.answer, "\n")
        for s in result.sources:
            print(f"[PMID:{s['pmid']}] {s['title']} ({s['journal']}, {s['year']}) {s['url']}")
        flagged = [c for c in result.checks if c.status not in ("supported", "partial")]
        if flagged:
            print(f"\n{len(flagged)} citation(s) could not be verified:")
            for c in flagged:
                print(f"  PMID {c.pmid}: {c.status} - {c.reason}")
        if result.uncited:
            print(f"\n{len(result.uncited)} statement(s) carry no citation; treat them with caution.")
        return

    from litagent.evaluate import evaluate, format_summary, load_questions

    questions = load_questions(args.questions)[: args.limit]
    judge = make_llm(llm.provider, args.judge_model or llm.model, args.base_url)
    summary = evaluate(questions, assistant, judge, args.out)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
