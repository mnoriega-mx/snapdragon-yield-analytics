"""
Command-line runner for the yield analytics agent.

Use this for quick manual smoke tests before the Streamlit UI is in place.

Examples:
    python -m agent.run "How many chips were produced today?"
    python -m agent.run --trace "Why did yield drop this afternoon?"
    python -m agent.run --question-file my_question.txt

The script needs ANTHROPIC_API_KEY set, either in the environment or in
a .env file at the project root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import run_agent


def _read_question(args: argparse.Namespace) -> str:
    if args.question_file:
        return Path(args.question_file).read_text(encoding="utf-8").strip()
    if args.question:
        return " ".join(args.question).strip()
    raise SystemExit("Provide a question as a positional argument or via --question-file.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Snapdragon yield analytics agent a question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The question to ask, in plain English.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Read the question from a file instead of the command line.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the full agent trace (tool calls, intermediate text) before the answer.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Safety cap on the number of agent loop iterations (default: 8).",
    )
    args = parser.parse_args(argv)

    question = _read_question(args)
    print(f"[question] {question}\n", flush=True)

    result = run_agent(question, max_iterations=args.max_iterations)

    if args.trace:
        print(result.render_trace())
        print()
    else:
        for step in result.trace:
            for call in step.tool_calls:
                print(f"[tool] {call['name']} -> {call['result_summary']}")
        if result.trace and any(step.tool_calls for step in result.trace):
            print()

    print(f"[answer] {result.answer}")
    print(
        f"\n[meta] iterations={result.iterations} "
        f"duration={result.total_duration_ms:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
