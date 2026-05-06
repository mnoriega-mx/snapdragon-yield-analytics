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

import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from . import tools as tool_catalog
from .prompts import SYSTEM_PROMPT

load_dotenv()


DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ITERATIONS = 8


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


@dataclass
class AgentResult:
    """Final return value of `run_agent`."""
    answer: str
    trace: list[TraceStep]
    iterations: int
    total_duration_ms: float

    def render_trace(self) -> str:
        """Format the trace as plain text for CLI output."""
        lines: list[str] = []
        for step in self.trace:
            lines.append(f"--- step {step.iteration} ({step.duration_ms:.0f} ms, stop={step.stop_reason}) ---")
            for text in step.text_blocks:
                lines.append(f"[claude] {text.strip()}")
            for call in step.tool_calls:
                args = call["input"]
                summary = call.get("result_summary", "")
                lines.append(f"[tool ] {call['name']}({_pretty_args(args)}) -> {summary}")
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
) -> AgentResult:
    """Send `question` to Claude and run the tool-use loop to completion.

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

    Returns:
        AgentResult with the final answer, the structured trace, and timing.
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

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": question},
    ]
    trace: list[TraceStep] = []
    started = time.time()

    for iteration in range(1, max_iterations + 1):
        step_start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tool_catalog.TOOL_SCHEMAS,
            messages=messages,
        )

        step = TraceStep(iteration=iteration, stop_reason=response.stop_reason)
        tool_uses: list[Any] = []
        assistant_content: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                step.text_blocks.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
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

        # If Claude is done, return the joined text.
        if response.stop_reason != "tool_use":
            step.duration_ms = (time.time() - step_start) * 1000
            trace.append(step)
            answer = "\n\n".join(step.text_blocks).strip()
            return AgentResult(
                answer=answer or "(no text returned)",
                trace=trace,
                iterations=iteration,
                total_duration_ms=(time.time() - started) * 1000,
            )

        # Otherwise execute each tool_use block and gather tool_result blocks.
        tool_result_blocks: list[dict[str, Any]] = []
        for tu in tool_uses:
            result = tool_catalog.execute_tool(tu.name, tu.input or {})
            summary = _summarize_tool_result(result)
            step.tool_calls.append({
                "id": tu.id,
                "name": tu.name,
                "input": tu.input,
                "result_summary": summary,
            })
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": tool_catalog.serialize_tool_result(result),
            })

        messages.append({"role": "user", "content": tool_result_blocks})
        step.duration_ms = (time.time() - step_start) * 1000
        trace.append(step)

    # Hit the iteration cap without a final answer.
    return AgentResult(
        answer=(
            "Agent exceeded the maximum number of iterations "
            f"({max_iterations}) without producing a final answer."
        ),
        trace=trace,
        iterations=max_iterations,
        total_duration_ms=(time.time() - started) * 1000,
    )
