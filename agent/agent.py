"""
Claude API agent loop for the Snapdragon Yield Analytics demo.

The loop is the standard Anthropic Messages tool-use pattern:

    1. Send the user's question to Claude with the tool catalog attached.
    2. Read the response.
       - If stop_reason == 'end_turn', we have a final answer; return it.
       - If stop_reason == 'tool_use', execute every tool_use block in
         order, append a tool_result content block per call, and resend
         the conversation.
    3. Repeat until Claude is done or until we hit a safety cap on the
       number of iterations.

The loop also records a structured trace of every step (tool calls, tool
results, assistant text). The Streamlit UI on Day 5 will render the
trace; the CLI runner on Day 2 just prints it.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from . import tools as tool_catalog
from .prompts import SYSTEM_PROMPT

# override=True so that an empty/stale shell value cannot mask the .env key.
load_dotenv(override=True)


DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ITERATIONS = 8

_log = logging.getLogger("snapdragon_agent.agent")


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """One iteration of the agent loop."""
    iteration: int
    text_blocks: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    duration_ms: float = 0.0
    # Token usage from response.usage. Cache fields are 0 when caching is
    # not in effect; non-zero values prove the cache breakpoints are
    # actually firing.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class AgentResult:
    """Final return value of `run_agent`."""
    answer: str
    trace: list[TraceStep]
    iterations: int
    total_duration_ms: float
    # Full conversation history at the end of the run, in Anthropic Messages
    # shape. A chat UI persists this and passes it back as the prefix of the
    # next turn so follow-up questions inherit context.
    messages: list[dict[str, Any]] = field(default_factory=list)

    def render_trace(self) -> str:
        """Format the trace as plain text for CLI output."""
        lines: list[str] = []
        for step in self.trace:
            lines.append(
                f"--- step {step.iteration} ({step.duration_ms:.0f} ms, "
                f"in={step.input_tokens}, out={step.output_tokens}, "
                f"cache_w={step.cache_creation_tokens}, cache_r={step.cache_read_tokens}, "
                f"stop={step.stop_reason}) ---"
            )
            for text in step.text_blocks:
                lines.append(f"[claude] {text.strip()}")
            for call in step.tool_calls:
                args = call["input"]
                summary = call.get("result_summary", "")
                lines.append(f"[tool ] {call['name']}({_pretty_args(args)}) -> {summary}")
        # Aggregate totals to make it obvious whether caching is actually firing.
        total_in = sum(s.input_tokens for s in self.trace)
        total_out = sum(s.output_tokens for s in self.trace)
        total_cw = sum(s.cache_creation_tokens for s in self.trace)
        total_cr = sum(s.cache_read_tokens for s in self.trace)
        cache_pct = 100 * total_cr / max(total_in + total_cr, 1)
        lines.append(
            f"--- totals: in={total_in}, out={total_out}, "
            f"cache_w={total_cw}, cache_r={total_cr}, "
            f"hit_rate={cache_pct:.1f} percent ---"
        )
        lines.append("--- final answer ---")
        lines.append(self.answer.strip())
        return "\n".join(lines)


def _pretty_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, list) and len(v) > 4:
            parts.append(f"{k}=[{len(v)} items]")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Prompt caching helpers
# ---------------------------------------------------------------------------
#
# Anthropic's rate limits cap input tokens per minute, but cached tokens do
# not count toward the limit, and cache reads cost only 10 percent of the
# normal input rate. The agent loop sends the same system prompt and tool
# definitions on every iteration plus a conversation that grows by one
# user/assistant pair each round. Both are perfect cache targets.
#
# Strategy: two cache breakpoints per request.
#   1. cache_control on the last tool, which caches the whole tools list
#      (and the system block before it) as a fixed, long-lived segment.
#   2. cache_control on the last content block of the last message, which
#      caches the conversation prefix so each new iteration only pays
#      full price for the new turn.
#
# The 1024-token-per-breakpoint minimum is comfortably exceeded by the
# tools list alone, so both breakpoints effectively cache.


def _tools_with_cache(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow copy of the tool schemas with cache_control on
    the last entry, so the system block and the entire tools list are
    cached as one segment."""
    if not schemas:
        return schemas
    out = [dict(s) for s in schemas]
    out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out


def _messages_with_cache(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow copy of the messages list with cache_control on
    the LAST content block of the LAST message, so the conversation
    prefix becomes cacheable for the next request."""
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    last = out[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        new_blocks = [dict(b) for b in content]
        new_blocks[-1] = {**new_blocks[-1], "cache_control": {"type": "ephemeral"}}
        last["content"] = new_blocks
    return out


def _summarize_tool_result(result: dict[str, Any]) -> str:
    """Compact one-line summary of a tool result for the trace."""
    if "error" in result:
        return f"error: {result['error']}"
    if result.get("query_type") == "summary":
        s = result.get("summary", {})
        return (
            f"summary total={s.get('total_chips')} "
            f"passed={s.get('passed')} failed={s.get('failed')} "
            f"yield={s.get('yield')}"
        )
    if "ucl" in result and "lcl" in result:
        return (
            f"spc {result.get('metric')} "
            f"mean={result.get('mean')} std={result.get('std')} "
            f"ooc={len(result.get('out_of_control', []))}"
        )
    if "anomalous_windows" in result:
        return (
            f"anomalies n={result.get('n_total')} "
            f"failed={result.get('n_failed')} "
            f"flagged_hours={len(result.get('anomalous_windows', []))}"
        )
    if "chart_type" in result and "filename" in result:
        return f"chart {result.get('chart_type')} -> {result.get('filename')}"
    if "report" in result and "n_findings" in result:
        return (
            f"report findings={result.get('n_findings')} "
            f"recs={result.get('n_recommendations')} "
            f"chars={result.get('char_count')}"
        )
    rc = result.get("row_count")
    truncated = result.get("truncated")
    suffix = " (truncated)" if truncated else ""
    return f"row_count={rc}{suffix}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    system_prompt: str = SYSTEM_PROMPT,
    on_step: Any | None = None,
) -> AgentResult:
    """Send `question` to Claude and run the tool-use loop to completion.

    Thin wrapper around `run_agent_stream` that drains the event stream
    and returns the final AgentResult. `on_step` is invoked for each
    completed TraceStep so existing UI code that relied on the callback
    keeps working.

    Args:
        question: The user's natural-language question.
        client: Optional pre-built `anthropic.Anthropic` client. Mostly
            useful for tests that want to inject a fake. When omitted we
            create one using ANTHROPIC_API_KEY from the environment.
        model: Claude model name. Defaults to claude-sonnet-4-6 or
            whatever ANTHROPIC_MODEL says.
        max_tokens: Per-response cap for Claude.
        max_iterations: Safety cap on the number of agent iterations.
            Each iteration is one Messages API call.
        system_prompt: System prompt to use. Override for tests or custom
            personas.
        on_step: Optional callable invoked with the completed TraceStep
            after each iteration.

    Returns:
        AgentResult with the final answer, the structured trace, and timing.
    """
    final: AgentResult | None = None
    for event in run_agent_stream(
        question,
        client=client,
        model=model,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
    ):
        if event["type"] == "step" and on_step is not None:
            on_step(event["step"])
        elif event["type"] == "final":
            final = event["result"]
    assert final is not None  # the stream always yields a final event
    return final


def run_agent_stream(
    question_or_messages: str | list[dict[str, Any]],
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    system_prompt: str = SYSTEM_PROMPT,
) -> Iterator[dict[str, Any]]:
    """Yield a sequence of events for one agent run, ending with `final`.

    Args:
        question_or_messages: Either a plain string (starts a fresh
            conversation with the string as the first user turn) OR a
            list of API-shaped messages from a prior conversation, with
            the new user turn already appended at the end. The list form
            powers the chat UI: pass the running history so the agent
            inherits all prior tool calls and assistant turns.

    Event shapes:
      {"type": "assistant_text", "text": str, "iteration": int}
        Prose the model emitted in this iteration. May frame upcoming
        tool calls or be the final answer.
      {"type": "tool_call", "call": dict, "iteration": int}
        Emitted after a tool finishes executing. The call dict matches
        the entries that land in TraceStep.tool_calls (id, name, input,
        result_summary, chart_path, report).
      {"type": "step", "step": TraceStep}
        Emitted at the end of each iteration. Lets callers stream
        per-iteration aggregates (token counts, durations).
      {"type": "final", "result": AgentResult}
        Always the last event. `result.messages` is the full updated
        history a chat UI should persist.
    """
    if client is None:
        # Lazy import so test environments that mock the client never
        # require the real anthropic package to be installed.
        import anthropic  # type: ignore

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and put your key there, or export the variable in your shell."
            )
        client = anthropic.Anthropic(api_key=api_key)

    if isinstance(question_or_messages, str):
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": question_or_messages},
        ]
        log_question = question_or_messages
    else:
        # Caller passed a full history; copy so we never mutate their list.
        messages = [dict(m) for m in question_or_messages]
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        content = last_user.get("content") if last_user else None
        log_question = content if isinstance(content, str) else "(continued conversation)"

    trace: list[TraceStep] = []
    started = time.time()

    _log.info(
        "run start question=%r model=%s max_iterations=%d",
        log_question, model, max_iterations,
    )

    for iteration in range(1, max_iterations + 1):
        step_start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=_tools_with_cache(tool_catalog.TOOL_SCHEMAS),
            messages=_messages_with_cache(messages),
        )

        step = TraceStep(iteration=iteration, stop_reason=response.stop_reason)
        usage = getattr(response, "usage", None)
        if usage is not None:
            step.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            step.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            step.cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            step.cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        tool_uses: list[Any] = []
        assistant_content: list[dict[str, Any]] = []
        text_events: list[str] = []

        for block in response.content:
            if block.type == "text":
                step.text_blocks.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
                if block.text and block.text.strip():
                    text_events.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Append the assistant turn (text plus any tool_use blocks).
        messages.append({"role": "assistant", "content": assistant_content})

        # Surface the model's prose for THIS iteration. On a non-terminal
        # turn this is the "I'll start by..." framing that precedes the
        # next tool call; on a terminal turn it's the final prose answer.
        # The chat UI suppresses the terminal-turn copy when a structured
        # report exists, so it is safe to always yield.
        for text in text_events:
            yield {
                "type": "assistant_text",
                "text": text,
                "iteration": iteration,
            }

        # If Claude is done, finalize and yield the result.
        if response.stop_reason != "tool_use":
            step.duration_ms = (time.time() - step_start) * 1000
            trace.append(step)
            yield {"type": "step", "step": step}

            answer = "\n\n".join(step.text_blocks).strip()
            total_ms = (time.time() - started) * 1000
            _log.info(
                "iteration=%d stop=%s duration_ms=%.0f "
                "tokens_in=%d tokens_out=%d cache_r=%d cache_w=%d",
                iteration, step.stop_reason, step.duration_ms,
                step.input_tokens, step.output_tokens,
                step.cache_read_tokens, step.cache_creation_tokens,
            )
            _log.info(
                "run end iterations=%d total_duration_ms=%.0f answer_chars=%d",
                iteration, total_ms, len(answer),
            )
            result = AgentResult(
                answer=answer or "(no text returned)",
                trace=trace,
                iterations=iteration,
                total_duration_ms=total_ms,
                messages=messages,
            )
            yield {"type": "final", "result": result}
            return

        # Otherwise execute each tool_use block, yielding tool_call events
        # as each one finishes so a UI can render them progressively.
        tool_result_blocks: list[dict[str, Any]] = []
        for tu in tool_uses:
            result = tool_catalog.execute_tool(tu.name, tu.input or {})
            summary = _summarize_tool_result(result)
            call_record = {
                "id": tu.id,
                "name": tu.name,
                "input": tu.input,
                "result_summary": summary,
                # Surface a couple of well-known artifacts so a UI can
                # pull them out without re-parsing the raw tool result.
                "chart_path": result.get("path") if tu.name == "generate_chart" else None,
                "report": result.get("report") if tu.name == "write_summary_report" else None,
            }
            step.tool_calls.append(call_record)
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": tool_catalog.serialize_tool_result(result),
            })
            yield {
                "type": "tool_call",
                "call": call_record,
                "iteration": iteration,
            }

        messages.append({"role": "user", "content": tool_result_blocks})
        step.duration_ms = (time.time() - step_start) * 1000
        trace.append(step)
        _log.info(
            "iteration=%d stop=%s duration_ms=%.0f "
            "tokens_in=%d tokens_out=%d cache_r=%d cache_w=%d "
            "tool_calls=%s",
            iteration, step.stop_reason, step.duration_ms,
            step.input_tokens, step.output_tokens,
            step.cache_read_tokens, step.cache_creation_tokens,
            [c["name"] for c in step.tool_calls],
        )
        yield {"type": "step", "step": step}

    # Hit the iteration cap without a final answer.
    total_ms = (time.time() - started) * 1000
    _log.warning(
        "run end iterations=%d total_duration_ms=%.0f cap_reached=true",
        max_iterations, total_ms,
    )
    result = AgentResult(
        answer=(
            "Agent exceeded the maximum number of iterations "
            f"({max_iterations}) without producing a final answer."
        ),
        trace=trace,
        iterations=max_iterations,
        total_duration_ms=total_ms,
        messages=messages,
    )
    yield {"type": "final", "result": result}
