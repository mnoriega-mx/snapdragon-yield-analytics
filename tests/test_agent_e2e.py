"""
End-to-end tests for the Claude API agent loop.

These tests do not hit the real Anthropic API. They drive `run_agent`
against a `FakeAnthropicClient` that returns scripted responses,
verifying the loop's mechanics: tool dispatch, trace recording,
artifact extraction, prompt-cache breakpoints, error handling, the
on_step callback, and the iteration safety cap.

The non-deterministic real-API scenario validation lives in
`scripts/validate_scenarios.py` and writes its results to
`docs/scenario_validation.md`. That script is run manually before a
demo, never as part of the unit-test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent import tools as tool_catalog
from agent.agent import run_agent


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


@dataclass
class FakeBlock:
    """Minimal stand-in for an Anthropic content block.

    The real SDK exposes attributes (block.type, block.text, block.id,
    block.name, block.input). The agent loop only reads those, so a
    plain dataclass is enough.
    """
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 30
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class _Messages:
    def __init__(self, parent: "FakeAnthropicClient") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> FakeResponse:
        self._parent.recorded_calls.append(kwargs)
        if not self._parent.scripted:
            raise AssertionError(
                "FakeAnthropicClient ran out of scripted responses; "
                "the test should script enough turns or reduce max_iterations."
            )
        return self._parent.scripted.pop(0)


class FakeAnthropicClient:
    """Stand-in for `anthropic.Anthropic`, driven by a queue of FakeResponses."""

    def __init__(self, scripted: list[FakeResponse]) -> None:
        self.scripted = list(scripted)
        self.recorded_calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)


def _text(t: str) -> FakeBlock:
    return FakeBlock(type="text", text=t)


def _tool_use(call_id: str, name: str, args: dict[str, Any]) -> FakeBlock:
    return FakeBlock(type="tool_use", id=call_id, name=name, input=args)


def _end_turn(*blocks: FakeBlock) -> FakeResponse:
    return FakeResponse(content=list(blocks), stop_reason="end_turn")


def _with_tool_calls(*blocks: FakeBlock) -> FakeResponse:
    return FakeResponse(content=list(blocks), stop_reason="tool_use")


# ---------------------------------------------------------------------------
# Helpers for poking at a result
# ---------------------------------------------------------------------------


def _all_tool_calls(result) -> list[dict[str, Any]]:
    return [c for step in result.trace for c in step.tool_calls]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_drives_yield_drop_scenario_to_completion(chip_db, tmp_path, monkeypatch):
    """A scripted yield investigation: 4 tool calls, then a final text turn."""
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)
    monkeypatch.setattr(tool_catalog, "DEFAULT_CHART_DIR", tmp_path)

    client = FakeAnthropicClient([
        _with_tool_calls(
            _text("Let me start with a summary."),
            _tool_use("c1", "query_database", {"query_type": "summary"}),
        ),
        _with_tool_calls(
            _text("Yield drops on the last day. Looking deeper."),
            _tool_use("c2", "detect_anomalies", {
                "start_time": "2026-04-07 00:00:00",
                "end_time": "2026-04-08 00:00:00",
            }),
        ),
        _with_tool_calls(
            _text("NPU is the suspect. Generating a chart."),
            _tool_use("c3", "generate_chart", {
                "chart_type": "correlation_chart",
                "primary_metric": "npu_tops",
                "secondary_metric": "npu_power_w",
                "start_time": "2026-04-07 00:00:00",
                "end_time": "2026-04-08 00:00:00",
            }),
        ),
        _with_tool_calls(
            _tool_use("c4", "write_summary_report", {
                "findings": [
                    {
                        "category": "NPU performance",
                        "description": (
                            "Yield in the afternoon (14:00 onward) is "
                            "dominated by chips with low NPU TOPS."
                        ),
                    }
                ],
                "root_cause_hypothesis": (
                    "Hexagon NPU power-domain excursion starting at 14:00."
                ),
                "recommendations": [
                    "Quarantine the afternoon-shift wafers.",
                ],
            }),
        ),
        _end_turn(_text("Investigation complete. Report attached above.")),
    ])

    result = run_agent(
        "Why did yield drop today?",
        client=client,
        max_iterations=8,
    )

    # Five iterations end-to-end: four tool turns plus the final text.
    assert result.iterations == 5

    tool_names = [c["name"] for c in _all_tool_calls(result)]
    assert tool_names == [
        "query_database",
        "detect_anomalies",
        "generate_chart",
        "write_summary_report",
    ]

    # The chart artifact is surfaced on the relevant trace entry and
    # the file actually exists on disk.
    chart_calls = [c for c in _all_tool_calls(result) if c["name"] == "generate_chart"]
    assert chart_calls and chart_calls[0]["chart_path"]
    assert Path(chart_calls[0]["chart_path"]).exists()

    # The report text is surfaced on the write_summary_report trace entry.
    report_calls = [c for c in _all_tool_calls(result) if c["name"] == "write_summary_report"]
    assert report_calls
    assert "Hexagon NPU" in report_calls[0]["report"]

    # Final acknowledgement text is the agent's answer.
    assert "complete" in result.answer.lower()


def test_agent_returns_safety_message_on_iteration_cap(chip_db, monkeypatch):
    """Agent stops at max_iterations and returns the safety message."""
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)

    client = FakeAnthropicClient([
        _with_tool_calls(_tool_use("c1", "query_database", {"query_type": "summary"})),
        _with_tool_calls(_tool_use("c2", "query_database", {"query_type": "summary"})),
    ])

    result = run_agent("loop forever", client=client, max_iterations=2)

    assert result.iterations == 2
    assert "exceeded the maximum" in result.answer.lower()


def test_agent_returns_tool_error_to_claude(chip_db, monkeypatch):
    """A tool failure is captured in the trace AND sent back to Claude."""
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)

    client = FakeAnthropicClient([
        _with_tool_calls(_tool_use("c1", "query_database", {"query_type": "bogus"})),
        _end_turn(_text("I tried, but the tool rejected my arguments.")),
    ])

    result = run_agent("test errors", client=client)

    first_call = result.trace[0].tool_calls[0]
    assert first_call["name"] == "query_database"
    assert "error" in first_call["result_summary"]

    # The second request to Claude carries a tool_result whose content
    # is the error string. This is what makes the agent self-correcting:
    # Claude can read the error and try a different approach.
    assert len(client.recorded_calls) == 2
    second_request_messages = client.recorded_calls[1]["messages"]
    user_msg = second_request_messages[-1]
    assert user_msg["role"] == "user"
    tool_result_block = user_msg["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert "must be one of" in tool_result_block["content"]


def test_agent_invokes_on_step_callback_per_iteration(chip_db, monkeypatch):
    """on_step fires exactly once per iteration, in order."""
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)

    client = FakeAnthropicClient([
        _with_tool_calls(_tool_use("c1", "query_database", {"query_type": "summary"})),
        _end_turn(_text("done.")),
    ])

    seen_iterations: list[int] = []
    seen_stops: list[str] = []

    def cb(step):
        seen_iterations.append(step.iteration)
        seen_stops.append(step.stop_reason)

    run_agent("test", client=client, on_step=cb)

    assert seen_iterations == [1, 2]
    assert seen_stops == ["tool_use", "end_turn"]


def test_agent_applies_prompt_cache_breakpoints(chip_db, monkeypatch):
    """cache_control sits on the LAST tool and the LAST message block.

    These breakpoints are how the loop keeps tier-1 token usage in check
    across multi-turn investigations. If they regress (extra cache markers,
    missing markers), this test will fail before a demo run melts down on
    rate limits.
    """
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)

    client = FakeAnthropicClient([
        _end_turn(_text("nothing to do.")),
    ])

    run_agent("hi", client=client)

    request = client.recorded_calls[0]
    tools_sent = request["tools"]
    assert tools_sent[-1].get("cache_control") == {"type": "ephemeral"}
    for tool in tools_sent[:-1]:
        assert "cache_control" not in tool

    last_msg = request["messages"][-1]
    last_block = last_msg["content"][-1]
    assert last_block.get("cache_control") == {"type": "ephemeral"}


def test_agent_records_token_usage_per_step(chip_db, monkeypatch):
    """The trace surfaces input/output/cache tokens from response.usage."""
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)

    response = _end_turn(_text("done."))
    response.usage = FakeUsage(
        input_tokens=1234,
        output_tokens=56,
        cache_creation_input_tokens=789,
        cache_read_input_tokens=10,
    )
    client = FakeAnthropicClient([response])

    result = run_agent("hi", client=client)

    step = result.trace[0]
    assert step.input_tokens == 1234
    assert step.output_tokens == 56
    assert step.cache_creation_tokens == 789
    assert step.cache_read_tokens == 10


def test_agent_handles_unknown_tool_name(chip_db, monkeypatch):
    """A made-up tool name is rejected by execute_tool and surfaced as an error."""
    monkeypatch.setattr(tool_catalog, "DEFAULT_DB_PATH", chip_db)

    client = FakeAnthropicClient([
        _with_tool_calls(_tool_use("c1", "definitely_not_a_tool", {})),
        _end_turn(_text("the tool I asked for did not exist.")),
    ])

    result = run_agent("test", client=client)

    first_call = result.trace[0].tool_calls[0]
    assert "unknown tool" in first_call["result_summary"]
