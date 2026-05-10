"""
Run the three brief scenarios end-to-end against the live agent.

This script hits the real Anthropic API. Use it before recording the
demo to confirm the agent handles the canonical questions sensibly.
The output is a markdown report saved to docs/scenario_validation.md
plus a per-scenario PASS/FAIL pretty-print on stdout.

Each scenario carries a short list of soft expectations: presence of
key concepts in the agent's text and report, and which tools were
called. Expectations are intentionally lenient (the LLM's exact
wording shifts run to run); they catch outright regressions like
"agent never calls write_summary_report on a yield-drop question".

Run:
    python scripts/validate_scenarios.py
    python scripts/validate_scenarios.py --scenario "Anomaly investigation"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import AgentResult, run_agent  # noqa: E402
from agent.logging_setup import setup_file_logging  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class ScenarioRun:
    name: str
    question: str
    answer: str
    report: str | None
    tool_names: list[str]
    chart_paths: list[str]
    iterations: int
    duration_ms: float


@dataclass
class Scenario:
    name: str
    question: str
    expectations: list[tuple[str, Callable[[ScenarioRun], bool]]] = field(default_factory=list)


def _has_concept(*concepts: str) -> Callable[[ScenarioRun], bool]:
    """Pass when at least one concept appears (case-insensitive) in answer or report."""
    def check(run: ScenarioRun) -> bool:
        haystack = (run.answer + (run.report or "")).lower()
        return any(c.lower() in haystack for c in concepts)
    return check


def _called_tool(tool_name: str) -> Callable[[ScenarioRun], bool]:
    return lambda run: tool_name in run.tool_names


def _called_at_least(n: int) -> Callable[[ScenarioRun], bool]:
    return lambda run: len(run.tool_names) >= n


def _did_not_call(tool_name: str) -> Callable[[ScenarioRun], bool]:
    return lambda run: tool_name not in run.tool_names


SCENARIOS: list[Scenario] = [
    Scenario(
        name="Normal operation",
        question="How is yield today?",
        expectations=[
            ("calls query_database", _called_tool("query_database")),
            ("does not start a deep investigation", _did_not_call("write_summary_report")),
            ("answer mentions yield", _has_concept("yield", "percent")),
        ],
    ),
    Scenario(
        name="Anomaly investigation",
        question="Why did yield drop today?",
        expectations=[
            ("calls at least 3 tools", _called_at_least(3)),
            ("calls detect_anomalies", _called_tool("detect_anomalies")),
            ("produces a structured report", _called_tool("write_summary_report")),
            ("references NPU", _has_concept("NPU", "Hexagon")),
            ("references the afternoon", _has_concept("afternoon", "14:00", "14:")),
        ],
    ),
    Scenario(
        name="Specific lookup",
        question="Show me Wafer W050's performance.",
        expectations=[
            ("calls query_database", _called_tool("query_database")),
            ("answer references W050", _has_concept("W050")),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Trace introspection helpers
# ---------------------------------------------------------------------------


def _final_report(result: AgentResult) -> str | None:
    """Return the most recent write_summary_report markdown, if any."""
    last: str | None = None
    for step in result.trace:
        for call in step.tool_calls:
            if call.get("name") == "write_summary_report" and call.get("report"):
                last = call["report"]
    return last


def _all_chart_paths(result: AgentResult) -> list[str]:
    return [
        call["chart_path"]
        for step in result.trace
        for call in step.tool_calls
        if call.get("name") == "generate_chart" and call.get("chart_path")
    ]


def _all_tool_names(result: AgentResult) -> list[str]:
    return [c["name"] for step in result.trace for c in step.tool_calls]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_scenario(scenario: Scenario) -> ScenarioRun:
    result = run_agent(scenario.question)
    return ScenarioRun(
        name=scenario.name,
        question=scenario.question,
        answer=result.answer,
        report=_final_report(result),
        tool_names=_all_tool_names(result),
        chart_paths=_all_chart_paths(result),
        iterations=result.iterations,
        duration_ms=result.total_duration_ms,
    )


def _evaluate(scenario: Scenario, run: ScenarioRun) -> list[tuple[str, bool]]:
    return [(label, fn(run)) for label, fn in scenario.expectations]


def render_markdown(
    items: list[tuple[Scenario, ScenarioRun, list[tuple[str, bool]]]],
) -> str:
    lines: list[str] = []
    lines.append("# Scenario validation report")
    lines.append("")
    lines.append(
        "Generated " + time.strftime("%Y-%m-%d %H:%M") +
        " by `python scripts/validate_scenarios.py`."
    )
    lines.append("")
    lines.append(
        "Each scenario runs the live agent against the canonical question "
        "from the project brief and checks a handful of soft expectations "
        "(concept presence, tools called). Wording shifts run to run; the "
        "checks are designed to catch outright regressions, not enforce "
        "exact strings."
    )
    lines.append("")

    for scen, run, results in items:
        passed = sum(1 for _, ok in results if ok)
        total = len(results)
        status = "PASS" if passed == total else "FAIL"
        lines.append(f"## {scen.name} -- {status} ({passed}/{total})")
        lines.append("")
        lines.append(f"**Question:** {scen.question}")
        lines.append("")
        lines.append(
            f"Iterations: {run.iterations}, "
            f"duration: {run.duration_ms:.0f} ms, "
            f"tools called: {', '.join(run.tool_names) or '(none)'}"
        )
        if run.chart_paths:
            lines.append("")
            lines.append(
                "Charts generated: " +
                ", ".join(Path(p).name for p in run.chart_paths)
            )
        lines.append("")
        for label, ok in results:
            mark = "x" if ok else " "
            lines.append(f"- [{mark}] {label}")
        lines.append("")
        lines.append("**Final answer**")
        lines.append("")
        lines.append("```")
        lines.append(run.answer.strip() or "(empty)")
        lines.append("```")
        if run.report:
            lines.append("")
            lines.append("**Report**")
            lines.append("")
            lines.append(run.report.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "scenario_validation.md",
        help="Where to write the markdown report.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[s.name for s in SCENARIOS],
        help="Run only this scenario (can repeat). Default: all three.",
    )
    args = parser.parse_args(argv)

    log_path = setup_file_logging()
    print(f"[log] {log_path}\n", flush=True)

    selected = (
        SCENARIOS
        if not args.scenario
        else [s for s in SCENARIOS if s.name in args.scenario]
    )

    rows: list[tuple[Scenario, ScenarioRun, list[tuple[str, bool]]]] = []
    overall_pass = True
    for scen in selected:
        print(f"=== {scen.name} ===")
        print(f"question: {scen.question}")
        run = run_scenario(scen)
        results = _evaluate(scen, run)
        passed = sum(1 for _, ok in results if ok)
        total = len(results)
        print(f"  iterations: {run.iterations}, duration: {run.duration_ms:.0f} ms")
        print(f"  tools: {', '.join(run.tool_names) or '(none)'}")
        for label, ok in results:
            print(f"  [{'x' if ok else ' '}] {label}")
        print(f"  -> {passed}/{total}\n", flush=True)
        if passed < total:
            overall_pass = False
        rows.append((scen, run, results))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Report written to {args.out}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
