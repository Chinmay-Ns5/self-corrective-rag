"""Run the backend from the repository root: python -m scripts.ask 'question'."""
import argparse
import json
from contextlib import redirect_stdout
import sys

from src.crag.service import make_pipeline
from src.crag.pipeline import clarification_result


def main():
    parser = argparse.ArgumentParser(description="Ask the self-corrective RAG backend")
    parser.add_argument("question")
    args = parser.parse_args()
    try:
        # Keep stdout machine-readable even while the database/model loads.
        with redirect_stdout(sys.stderr):
            if not args.question.strip():
                raise ValueError("Question must not be empty")
            result = clarification_result(args.question) or make_pipeline().ask(args.question)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
