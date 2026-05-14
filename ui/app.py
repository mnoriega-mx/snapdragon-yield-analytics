"""
Streamlit web UI for the Snapdragon Yield Analytics agent.

Run with:
    streamlit run ui/app.py

The app gives a yield engineer a question box, runs the Claude agent,
streams every tool call into a status panel as it happens, and finally
displays any generated charts inline, the structured markdown report,
and the agent's prose answer.

Requires ANTHROPIC_API_KEY set in the environment or in a .env file at
the project root.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Make the agent package importable regardless of how Streamlit is launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import run_agent_stream  # noqa: E402
from agent.logging_setup import setup_file_logging  # noqa: E402

# One log file per Streamlit process; idempotent so reruns reuse it.
setup_file_logging()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Snapdragon Yield Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Design system: theme tokens, custom CSS, branded shell
# ---------------------------------------------------------------------------

# Shared color tokens used by both CSS and Altair charts. Keeping them in one
# place is what makes the dashboard, charts, and report read as one product.
COLORS = {
    "bg": "#F5F7FA",
    "panel": "#FFFFFF",
    "panel_2": "#EEF2F8",
    "border": "rgba(15, 23, 42, 0.08)",
    "border_strong": "rgba(15, 23, 42, 0.18)",
    "text": "#0B1220",
    "muted": "#5B6577",
    "accent": "#3253DC",        # Qualcomm blue, primary
    "accent_dark": "#1F3CA8",   # Darker variant for gradients
    "cyan": "#1B8BB8",          # Secondary line color, darker for light bg
    "success": "#1FA86A",       # Darker green for white-background contrast
    "warning": "#C97A0E",       # Darker amber
    "danger": "#C8261C",        # Darker red
    "grid": "rgba(15, 23, 42, 0.07)",
}


CUSTOM_CSS = f"""
<style>
/* === Page chrome === */
.block-container {{
  /* Top padding is generous so our page header clears the Streamlit
     Cloud toolbar (Share / Star / GitHub / kebab) that overlays the
     top-right corner on deployed apps. */
  padding-top: 4.5rem !important;
  padding-bottom: 3rem !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  max-width: 1320px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}}
.stApp {{ background: {COLORS["bg"]}; font-size: 16px; color: {COLORS["text"]}; }}

/* Streamlit Cloud's top toolbar icons (Share / Star / GitHub / kebab)
   inherit the original dark theme color. We only set `color` on the
   wrappers so that SVG paths using fill="currentColor" or
   stroke="currentColor" inherit our text color. Setting fill/stroke
   directly on path/svg backfires: Streamlit's kebab icon has a
   transparent background path with fill="none", and a blanket fill
   override turns the whole icon into a solid dark square. */
header[data-testid="stHeader"],
header[data-testid="stHeader"] [data-testid="stToolbar"] button,
header[data-testid="stHeader"] [data-testid="stToolbar"] a,
header[data-testid="stHeader"] [data-testid="stToolbar"] svg,
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] button svg {{
  color: {COLORS["text"]} !important;
}}
header[data-testid="stHeader"] [data-testid="stToolbar"] button:hover,
header[data-testid="stHeader"] [data-testid="stToolbar"] a:hover {{
  color: {COLORS["accent"]} !important;
  background: rgba(50, 83, 220, 0.08) !important;
}}
header[data-testid="stHeader"] [data-testid="stToolbar"] button:hover svg,
header[data-testid="stHeader"] [data-testid="stToolbar"] a:hover svg {{
  color: {COLORS["accent"]} !important;
}}
/* Only cover prose-bearing elements. Avoid blanket-coloring spans/labels
   because the status pill and delta chip use spans whose colors are
   status-driven (red ANOMALOUS, green up, etc). */
.stApp p, .stApp li,
.stApp .stMarkdown, .stApp [data-testid="stMarkdownContainer"] {{
  color: {COLORS["text"]};
}}

/* Hide the default Streamlit header chrome for a cleaner look */
header[data-testid="stHeader"] {{ background: transparent; }}

/* === Branded header === */
.snd-header {{
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 26px 30px;
  background: {COLORS["panel"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 16px;
  margin: 0 0 32px 0;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  position: relative;
  overflow: hidden;
}}
.snd-header::before {{
  /* A vivid Qualcomm-blue accent bar along the left edge. */
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  background: linear-gradient(180deg, {COLORS["accent"]} 0%, {COLORS["accent_dark"]} 100%);
}}
.snd-mark {{
  width: 50px; height: 50px;
  display: grid; place-items: center;
  background: linear-gradient(135deg, {COLORS["accent"]} 0%, {COLORS["accent_dark"]} 100%);
  border-radius: 12px;
  font-weight: 800; font-size: 19px;
  color: white; letter-spacing: 0.02em;
  box-shadow: 0 10px 26px rgba(50, 83, 220, 0.38);
}}
.snd-title-block {{ display: flex; flex-direction: column; }}
.snd-title {{
  font-size: 1.7rem; font-weight: 700;
  color: {COLORS["text"]}; letter-spacing: -0.015em; line-height: 1.15;
}}
.snd-subtitle {{
  font-size: 0.98rem; color: {COLORS["muted"]}; margin-top: 4px;
}}
.snd-spacer {{ flex: 1; }}
.snd-tag {{
  font-size: 0.78rem; font-weight: 600;
  color: {COLORS["accent"]};
  background: rgba(50, 83, 220, 0.08);
  border: 1px solid rgba(50, 83, 220, 0.30);
  padding: 6px 14px; border-radius: 999px;
  letter-spacing: 0.08em; text-transform: uppercase;
}}

/* === Section labels === */
.snd-section {{
  font-size: 0.82rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: {COLORS["accent"]};
  margin: 10px 0 16px 0;
}}

/* === KPI cards === */
.snd-kpi-grid {{
  display: grid;
  grid-template-columns: 1.6fr 1fr 1fr 1fr;
  gap: 14px;
  margin-bottom: 26px;
}}
.snd-card {{
  background: {COLORS["panel"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 16px;
  padding: 22px 24px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
.snd-card-hero::before {{
  /* Status-driven accent stripe along the top of the hero card. The
     color is set by snd-card-hero-{{ok|warn|alert}} modifiers so the
     stripe agrees with the status pill in the top-right corner. */
  content: "";
  position: absolute;
  left: 0; right: 0; top: 0;
  height: 3px;
  background: linear-gradient(90deg, {COLORS["accent"]} 0%, transparent 80%);
}}
.snd-card-hero-ok::before    {{ background: linear-gradient(90deg, {COLORS["success"]} 0%, transparent 80%); }}
.snd-card-hero-warn::before  {{ background: linear-gradient(90deg, {COLORS["warning"]} 0%, transparent 80%); }}
.snd-card-hero-alert::before {{ background: linear-gradient(90deg, {COLORS["danger"]}  0%, transparent 80%); }}
.snd-card-label {{
  font-size: 0.78rem; font-weight: 600;
  color: {COLORS["muted"]};
  letter-spacing: 0.08em; text-transform: uppercase;
  margin-bottom: 10px;
}}
.snd-card-value {{
  font-size: 2.3rem; font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: {COLORS["text"]};
  line-height: 1.1;
  display: flex; align-items: baseline; gap: 8px;
  flex-wrap: wrap;
}}
.snd-card-hero .snd-card-value {{ font-size: 3rem; }}
.snd-card-value-sm {{ font-size: 1.9rem !important; line-height: 1.2; }}
.snd-card-sub {{
  margin-top: 10px;
  font-size: 0.88rem;
  color: {COLORS["muted"]};
  font-variant-numeric: tabular-nums;
}}
.snd-delta-chip {{
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.82rem; font-weight: 600;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
}}
.snd-delta-up   {{ background: rgba(31, 168, 106, 0.14); color: {COLORS["success"]}; }}
.snd-delta-down {{ background: rgba(200, 38, 28, 0.14);  color: {COLORS["danger"]}; }}
.snd-delta-flat {{ background: rgba(91, 101, 119, 0.14); color: {COLORS["muted"]}; }}

/* Status pill, top-right corner of the hero card */
.snd-status {{
  position: absolute;
  top: 18px; right: 20px;
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 0.78rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 6px 12px; border-radius: 999px;
}}
.snd-status::before {{
  content: ""; width: 8px; height: 8px; border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 10px currentColor;
}}
.snd-status-ok    {{ background: rgba(31, 168, 106, 0.14); color: {COLORS["success"]}; }}
.snd-status-warn  {{ background: rgba(201, 122, 14, 0.14); color: {COLORS["warning"]}; }}
.snd-status-alert {{ background: rgba(200, 38, 28, 0.14);  color: {COLORS["danger"]}; }}

/* === Sample-question chips === */
/* Streamlit secondary buttons are restyled as compact chips. The primary
   Run button keeps its own treatment further down. */
div[data-testid="stButton"] > button[kind="secondary"] {{
  background: {COLORS["panel"]} !important;
  border: 1px solid {COLORS["border_strong"]} !important;
  color: {COLORS["text"]} !important;
  font-weight: 500 !important;
  border-radius: 999px !important;
  padding: 10px 20px !important;
  font-size: 0.95rem !important;
  min-height: 0 !important;
  height: auto !important;
  width: 100% !important;
  text-align: center !important;
  transition: all 0.15s ease !important;
}}
div[data-testid="stButton"] > button[kind="secondary"]:hover {{
  border-color: {COLORS["accent"]} !important;
  background: rgba(50, 83, 220, 0.10) !important;
  color: {COLORS["accent_dark"]} !important;
  box-shadow: 0 2px 8px rgba(50, 83, 220, 0.15) !important;
}}

/* === Primary Run button === */
div[data-testid="stButton"] > button[kind="primary"] {{
  background: linear-gradient(135deg, {COLORS["accent"]} 0%, {COLORS["accent_dark"]} 100%) !important;
  border: 0 !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  letter-spacing: 0.02em;
  box-shadow: 0 6px 18px rgba(50, 83, 220, 0.35) !important;
  padding: 13px 28px !important;
  border-radius: 12px !important;
}}
div[data-testid="stButton"] > button[kind="primary"]:hover {{
  filter: brightness(1.08);
  transform: translateY(-1px);
}}
div[data-testid="stButton"] > button[kind="primary"]:disabled {{
  background: {COLORS["panel_2"]} !important;
  box-shadow: none !important;
  color: {COLORS["muted"]} !important;
  transform: none;
}}

/* === Text input === */
[data-testid="stTextInput"] input {{
  background: {COLORS["panel"]} !important;
  border: 1px solid {COLORS["border_strong"]} !important;
  color: {COLORS["text"]} !important;
  font-size: 1.05rem !important;
  padding: 14px 18px !important;
  border-radius: 12px !important;
}}
[data-testid="stTextInput"] input:focus {{
  border-color: {COLORS["accent"]} !important;
  box-shadow: 0 0 0 3px rgba(50, 83, 220, 0.22) !important;
}}
[data-testid="stTextInput"] label {{
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: {COLORS["muted"]} !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}

/* === Live trace status panel === */
[data-testid="stStatusWidget"], div[data-testid="stStatus"] {{
  background: {COLORS["panel"]} !important;
  border: 1px solid {COLORS["border"]} !important;
  border-radius: 12px !important;
}}

/* === Expander === */
/* Streamlit's expander has separate header (summary) and content panes,
   each with a dark-theme bg baked into emotion CSS. Force both light. */
[data-testid="stExpander"] {{
  background: {COLORS["panel"]} !important;
  border: 1px solid {COLORS["border"]} !important;
  border-radius: 12px !important;
  overflow: hidden;
}}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] details > div {{
  background: {COLORS["panel"]} !important;
  color: {COLORS["text"]} !important;
}}
[data-testid="stExpander"] summary {{
  font-weight: 600;
  color: {COLORS["text"]} !important;
}}
[data-testid="stExpander"] svg {{
  fill: {COLORS["text"]} !important;
}}

/* === Dividers softer, with a touch of red on the long ones === */
hr {{
  border: 0 !important;
  height: 1px !important;
  background: {COLORS["border"]} !important;
  margin: 1.6rem 0 !important;
}}

/* === Code blocks in trace === */
[data-testid="stCodeBlock"] {{
  background: {COLORS["panel_2"]} !important;
  border: 1px solid {COLORS["border"]};
  border-radius: 8px;
}}

/* === Chat: query label, thinking indicator, trace lines === */
.snd-query-label {{
  display: flex; align-items: baseline; gap: 12px;
  padding: 6px 0 14px 0;
  flex-wrap: wrap;
}}
.snd-query-arrow {{
  font-size: 0.72rem; font-weight: 700;
  color: {COLORS["accent"]};
  letter-spacing: 0.1em; text-transform: uppercase;
}}
.snd-query-text {{
  font-size: 1.05rem;
  color: {COLORS["text"]};
  line-height: 1.4;
}}

.snd-thinking {{
  display: inline-flex; align-items: center; gap: 10px;
  padding: 10px 0;
  color: {COLORS["accent"]};
  font-size: 0.78rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.1em;
}}
.snd-thinking-dots {{ display: inline-flex; gap: 4px; }}
.snd-thinking-dots span {{
  width: 6px; height: 6px; border-radius: 50%;
  background: {COLORS["accent"]};
  box-shadow: 0 0 8px {COLORS["accent"]};
  animation: snd-thinking-bounce 1.2s ease-in-out infinite;
}}
.snd-thinking-dots span:nth-child(2) {{ animation-delay: 0.18s; }}
.snd-thinking-dots span:nth-child(3) {{ animation-delay: 0.36s; }}
@keyframes snd-thinking-bounce {{
  0%, 80%, 100% {{ opacity: 0.25; transform: translateY(0) scale(0.85); }}
  40% {{ opacity: 1; transform: translateY(-4px) scale(1.1); }}
}}

.snd-trace-line {{
  padding: 4px 0 4px 14px;
  margin: 4px 0;
  border-left: 2px solid {COLORS["border_strong"]};
  font-size: 0.95rem;
  line-height: 1.5;
}}
.snd-trace-line:hover {{ border-left-color: {COLORS["accent"]}; }}
.snd-trace-action {{ color: {COLORS["text"]}; font-weight: 500; }}
.snd-trace-hint {{
  color: {COLORS["muted"]};
  font-size: 0.85rem;
  margin-left: 6px;
  font-variant-numeric: tabular-nums;
}}
.snd-trace-hint.snd-trace-error {{ color: {COLORS["danger"]}; }}

/* === Hide the real Streamlit Stop/Clear buttons (and their anchor
   markers) off-screen. They stay clickable so the JS-injected twins
   inside the chat_input can forward clicks here. Descendant combinator
   (no `>`) is intentional, Streamlit wraps the marker in
   stMarkdownContainer which breaks a strict path. === */
[data-testid="stElementContainer"]:has(span.snd-stop-anchor),
[data-testid="stElementContainer"]:has(span.snd-stop-anchor) + [data-testid="stElementContainer"],
[data-testid="stElementContainer"]:has(span.snd-clear-anchor),
[data-testid="stElementContainer"]:has(span.snd-clear-anchor) + [data-testid="stElementContainer"] {{
  position: absolute !important;
  left: -10000px !important;
  top: auto !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}}

/* === Chat input: match the main block-container width === */
/* st.chat_input is pinned at the bottom by Streamlit, in its own fixed
   container that does NOT inherit .block-container's max-width. Without
   these overrides the input would span the full viewport and feel
   disconnected from the page content above. */
[data-testid="stBottom"] {{
  background: transparent !important;
}}
[data-testid="stBottom"] > div {{
  max-width: 1320px !important;
  min-width: 0 !important;
  width: 100% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
  /* Streamlit bakes a dark navy into this wrapper via its theme. Force
     it to inherit the page background so the chat strip blends with
     the rest of the light theme. */
  background: {COLORS["bg"]} !important;
}}
/* The inner stBottomBlockContainer ships with its own 1rem horizontal
   padding which stacks on top of the parent above, leaving the chat
   input visibly narrower than .block-container. Zero it out so the
   input lines up edge-to-edge with the page content. */
[data-testid="stBottomBlockContainer"],
.stBottomBlockContainer {{
  max-width: none !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
}}

[data-testid="stChatInput"] {{
  background: {COLORS["panel"]} !important;
  /* Solid, high-contrast border so anti-aliasing renders evenly all
     the way around. A low-contrast / low-alpha border looks dimmer on
     the curved corners than on the straight edges, which reads as
     "faded corners" on dark backgrounds. */
  border: 1px solid {COLORS["border_strong"]} !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
}}
[data-testid="stChatInput"]:focus-within {{
  border-color: {COLORS["accent"]} !important;
  box-shadow: 0 0 0 3px rgba(50, 83, 220, 0.22) !important;
}}
[data-testid="stChatInputSubmitButton"] {{
  color: {COLORS["accent"]} !important;
  background: rgba(50, 83, 220, 0.10) !important;
  border-radius: 8px !important;
  transition: all 0.15s ease;
}}
[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {{
  background: rgba(50, 83, 220, 0.20) !important;
  color: {COLORS["accent_dark"]} !important;
}}
[data-testid="stChatInputSubmitButton"]:disabled {{
  color: {COLORS["muted"]} !important;
  background: transparent !important;
}}
[data-testid="stChatInputSubmitButton"] svg {{
  fill: currentColor !important;
}}
/* The inner wrappers (and Streamlit's BaseUI textarea container) ship
   with their own dark backgrounds + border-radii that peek out behind
   the outer border at the rounded corners. Flatten them all. */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div {{
  background-color: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}}
[data-testid="stChatInput"] textarea {{
  background: transparent !important;
  color: {COLORS["text"]} !important;
  font-size: 1.02rem !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
  color: {COLORS["muted"]} !important;
  opacity: 1 !important;
}}

/* === Markdown report polish === */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
  font-size: 1rem;
  line-height: 1.65;
}}
[data-testid="stMarkdownContainer"] h2 {{
  color: {COLORS["text"]};
  font-size: 1.35rem !important;
  letter-spacing: -0.01em;
  margin-top: 1.8rem !important;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid {COLORS["border"]};
}}
[data-testid="stMarkdownContainer"] h3 {{
  color: {COLORS["text"]};
  font-size: 1.1rem !important;
  margin-top: 1.4rem !important;
}}
[data-testid="stMarkdownContainer"] strong {{ color: {COLORS["text"]}; }}
[data-testid="stMarkdownContainer"] code {{
  background: rgba(50, 83, 220, 0.14);
  color: #B6C5FF;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
}}
</style>
"""


def _inject_css() -> None:
    """Apply the design system on every rerun. Idempotent."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_header() -> None:
    """Branded header lockup that anchors the page."""
    st.markdown(
        """
        <div class="snd-header">
          <div class="snd-title-block">
            <div class="snd-title">Snapdragon Yield Analytics</div>
            <div class="snd-subtitle">AI agent for Hexagon NPU yield investigation on Snapdragon SoCs</div>
          </div>
          <div class="snd-spacer"></div>
          <span class="snd-tag">Synthetic data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    "How is yield today?",
    "Why did yield drop this afternoon?",
    "Are there anomalies in NPU performance today?",
    "Show me Wafer W050's performance.",
]


def _pretty_args(args: dict[str, Any] | None) -> str:
    """Render tool input args compactly for the trace lines."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if isinstance(v, list) and len(v) > 4:
            parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, str) and len(v) > 30:
            parts.append(f"{k}={v[:27]!r}...")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


TOOL_LABELS: dict[str, str] = {
    "query_database": "Reading production data",
    "calculate_spc_metrics": "Computing SPC control limits",
    "detect_anomalies": "Detecting anomalous hours",
    "generate_chart": "Generating chart",
    "write_summary_report": "Writing structured report",
}


def _friendly_tool_label(call: dict[str, Any]) -> str:
    """Turn a raw tool_call dict into a human-readable progress line."""
    name = call.get("name", "")
    base = TOOL_LABELS.get(name, name)
    args = call.get("input") or {}

    if name == "generate_chart":
        chart_type = (args.get("chart_type") or "").replace("_", " ")
        primary = args.get("primary_metric")
        secondary = args.get("secondary_metric")
        if chart_type == "correlation chart" and primary and secondary:
            return f"Generating correlation chart for {primary} vs {secondary}"
        if primary:
            return f"Generating {chart_type} for {primary}"
        if chart_type:
            return f"Generating {chart_type}"
        return base
    if name == "calculate_spc_metrics":
        metric = args.get("metric")
        if metric:
            return f"Computing SPC control limits for {metric}"
        return base
    if name == "detect_anomalies":
        return "Detecting anomalous hours and ranking metrics by correlation"
    return base


def _render_trace_line(call: dict[str, Any]) -> None:
    """One styled trace line: friendly action, optional dim result hint.

    Mirrors the conversation log style: an action verb on its own line, a
    short dim-text hint after the arrow when the tool returned something
    worth surfacing inline. The full raw payload still lives in the
    Thinking Breakdown expander, this is the scannable summary.
    """
    label = _friendly_tool_label(call)
    summary = call.get("result_summary") or ""

    if summary.startswith("error"):
        st.markdown(
            f'<div class="snd-trace-line">'
            f'<span class="snd-trace-action">{label}</span>'
            f'<span class="snd-trace-hint snd-trace-error">  ->  {summary}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    if summary:
        hint = summary if len(summary) < 110 else summary[:107] + "..."
        st.markdown(
            f'<div class="snd-trace-line">'
            f'<span class="snd-trace-action">{label}</span>'
            f'<span class="snd-trace-hint">  ->  {hint}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="snd-trace-line">'
        f'<span class="snd-trace-action">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

FAILURE_REASON_LABELS: dict[str, str] = {
    "npu_tops_below_spec": "NPU TOPS low",
    "npu_power_above_spec": "NPU power high",
    "cpu_freq_below_spec": "CPU freq low",
    "memory_bandwidth_low": "Memory BW low",
    "die_temp_over_threshold": "Die temp high",
}


def _friendly_failure_reason(raw: str | None) -> str:
    if not raw:
        return "none"
    return FAILURE_REASON_LABELS.get(raw, raw.replace("_", " ").capitalize())


@st.cache_data(ttl=60, show_spinner=False)
def _load_dashboard_data() -> dict[str, Any] | None:
    """Fetch today's stats and the multi-day breakdown for the dashboard.

    Cached for one minute so a rapid sequence of reruns does not hammer
    the database. Returns None if the database file is missing.
    """
    from agent.tools import query_database

    try:
        week = query_database(query_type="summary")["summary"]
    except FileNotFoundError:
        return None

    per_day = week.get("daily_yield") or []
    if not per_day:
        return {"error": "no data in database"}

    today_date = per_day[-1]["date"]  # most recent day in the window
    next_date = (
        datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    today = query_database(
        query_type="summary",
        start_time=f"{today_date} 00:00:00",
        end_time=f"{next_date} 00:00:00",
    )["summary"]

    today_hourly = today.get("hourly_yield", [])

    prior_days = per_day[:-1]
    if prior_days:
        prior_total = sum(d["n"] for d in prior_days)
        prior_passed = sum(d["passed"] for d in prior_days)
        prior_yield = prior_passed / prior_total if prior_total else 0.0
    else:
        prior_yield = 0.0

    return {
        "today_date": today_date,
        "today": today,
        "today_hourly": today_hourly,
        "per_day": per_day,
        "prior_yield": prior_yield,
    }


def _status_pill(yield_val: float) -> tuple[str, str]:
    """Return (css_class, label) for the dashboard status pill."""
    if yield_val < 0.85:
        return ("snd-status-alert", "ANOMALOUS")
    if yield_val < 0.93:
        return ("snd-status-warn", "CAUTION")
    return ("snd-status-ok", "HEALTHY")


def _delta_chip(delta_pp: float) -> str:
    """HTML chip for a percentage-point delta on yield."""
    if delta_pp > 0.05:
        cls, arrow = "snd-delta-up", "&#9650;"
    elif delta_pp < -0.05:
        cls, arrow = "snd-delta-down", "&#9660;"
    else:
        cls, arrow = "snd-delta-flat", "&#8226;"
    return f'<span class="snd-delta-chip {cls}">{arrow} {delta_pp:+.1f} pp</span>'


def _themed_axis(title: str) -> alt.Axis:
    """Altair axis styling tuned for the dark theme."""
    return alt.Axis(
        title=title,
        labelColor=COLORS["muted"],
        titleColor=COLORS["muted"],
        domainColor=COLORS["border_strong"],
        tickColor=COLORS["border_strong"],
        gridColor=COLORS["grid"],
        titleFontSize=11,
        labelFontSize=11,
    )


def _render_dashboard(data: dict[str, Any] | None) -> None:
    """Render the production-overview panel above the question box."""
    if data is None:
        st.warning(
            "Dashboard unavailable: database not loaded. "
            "Run `./venv/bin/python data/generate_data.py && "
            "./venv/bin/python data/setup_database.py`."
        )
        return
    if "error" in data:
        st.warning(f"Dashboard unavailable: {data['error']}")
        return

    today = data["today"]
    today_yield = today["yield"]
    prior_yield = data["prior_yield"]
    delta_pp = (today_yield - prior_yield) * 100  # percentage points

    status_cls, status_label = _status_pill(today_yield)
    fb = today.get("failure_breakdown") or []
    top_failure = _friendly_failure_reason(fb[0]["failure_reason"] if fb else None)
    n_prior = len(data["per_day"]) - 1
    fail_pct = (
        today["failed"] / today["total_chips"] * 100 if today["total_chips"] else 0.0
    )
    prior_caption = (
        f"vs prior {n_prior}-day avg {prior_yield:.1%}"
        if n_prior > 0
        else "no prior-day comparison"
    )

    st.markdown(
        f'<div class="snd-section">Today\'s production &middot; {data["today_date"]}</div>',
        unsafe_allow_html=True,
    )

    hero_stripe_cls = status_cls.replace("snd-status-", "snd-card-hero-")
    kpi_html = f"""
    <div class="snd-kpi-grid">
      <div class="snd-card snd-card-hero {hero_stripe_cls}">
        <div class="snd-card-label">Yield</div>
        <div class="snd-card-value">{today_yield:.1%}{_delta_chip(delta_pp)}</div>
        <div class="snd-card-sub">{prior_caption}</div>
        <span class="snd-status {status_cls}">{status_label}</span>
      </div>
      <div class="snd-card">
        <div class="snd-card-label">Chips tested</div>
        <div class="snd-card-value">{today["total_chips"]:,}</div>
        <div class="snd-card-sub">today</div>
      </div>
      <div class="snd-card">
        <div class="snd-card-label">Failures</div>
        <div class="snd-card-value">{today["failed"]:,}</div>
        <div class="snd-card-sub">{fail_pct:.1f}% of tested</div>
      </div>
      <div class="snd-card">
        <div class="snd-card-label">Top failure mode</div>
        <div class="snd-card-value snd-card-value-sm">{top_failure}</div>
      </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    if data["today_hourly"]:
        st.markdown(
            '<div class="snd-section">24-hour yield trend</div>',
            unsafe_allow_html=True,
        )
        hourly_df = pd.DataFrame(data["today_hourly"])
        hourly_df["hour"] = hourly_df["hour"].str[-2:]
        hourly_df["yield_pct"] = hourly_df["yield"] * 100

        trend = (
            alt.Chart(hourly_df)
            .mark_line(
                color=COLORS["accent"],
                strokeWidth=2.5,
                point=alt.OverlayMarkDef(filled=True, size=55, color=COLORS["accent"]),
            )
            .encode(
                x=alt.X("hour:O", axis=_themed_axis("Hour of day")),
                y=alt.Y(
                    "yield_pct:Q",
                    axis=_themed_axis("Yield (%)"),
                    scale=alt.Scale(zero=False, padding=10),
                ),
                tooltip=[
                    alt.Tooltip("hour:O", title="Hour"),
                    alt.Tooltip("yield_pct:Q", title="Yield %", format=".1f"),
                ],
            )
            .properties(height=260, background="transparent")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(trend, use_container_width=True)


# ---------------------------------------------------------------------------
# Native chart rendering (replaces matplotlib PNG embedding)
# ---------------------------------------------------------------------------

DB_PATH = PROJECT_ROOT / "data" / "chip_production.db"

METRIC_LABELS: dict[str, str] = {
    "npu_tops": "NPU TOPS",
    "npu_power_w": "NPU power (W)",
    "cpu_freq_ghz": "CPU frequency (GHz)",
    "memory_bandwidth_gbps": "Memory bandwidth (GB/s)",
    "die_temp_c": "Die temperature (C)",
}

CHART_TYPE_LABELS: dict[str, str] = {
    "spc_chart": "SPC chart",
    "correlation_chart": "Correlation",
    "failure_timeline": "Failure timeline",
}


@contextmanager
def _ro_connection():
    """Read-only SQLite connection mirroring the agent's `_connect` pattern."""
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


@st.cache_data(ttl=60, show_spinner=False)
def _query_hourly_metrics(start_time: str, end_time: str) -> pd.DataFrame:
    """Per-hour means of every metric, plus n and failure count, for the window."""
    sql = """
        SELECT
            substr(timestamp, 1, 13) AS hour,
            AVG(npu_tops) AS npu_tops,
            AVG(npu_power_w) AS npu_power_w,
            AVG(cpu_freq_ghz) AS cpu_freq_ghz,
            AVG(memory_bandwidth_gbps) AS memory_bandwidth_gbps,
            AVG(die_temp_c) AS die_temp_c,
            COUNT(*) AS n,
            SUM(CASE WHEN test_result = 'FAIL' THEN 1 ELSE 0 END) AS failures
        FROM chip_production_data
        WHERE timestamp >= ? AND timestamp < ?
        GROUP BY hour
        ORDER BY hour
    """
    with _ro_connection() as conn:
        df = pd.read_sql_query(sql, conn, params=(start_time, end_time))
    if not df.empty:
        df["short_hour"] = df["hour"].str[-2:]
    return df


def _render_spc_native(metric: str, start_time: str, end_time: str, label: str) -> None:
    """SPC chart: hourly mean line plus mean / UCL / LCL reference lines.

    Built with Altair so we can apply the dark-theme palette consistently
    (Snapdragon-red data line, dimmed white reference lines for mean and
    +/- 3 sigma control limits, sparse grid).
    """
    from agent.tools import calculate_spc_metrics

    df = _query_hourly_metrics(start_time, end_time)
    if df.empty:
        st.info("No data in the selected window.")
        return

    spc = calculate_spc_metrics(
        metric=metric,
        start_time=start_time,
        end_time=end_time,
        group_by="hour",
    )

    metric_label = METRIC_LABELS.get(metric, metric)
    plot_df = pd.DataFrame(
        {
            "hour": df["short_hour"],
            "value": df[metric].astype(float),
        }
    )

    mean_val = float(spc["mean"])
    ucl = float(spc["ucl"])
    lcl = float(spc["lcl"])

    line = (
        alt.Chart(plot_df)
        .mark_line(
            color=COLORS["accent"],
            strokeWidth=2.5,
            point=alt.OverlayMarkDef(filled=True, size=55, color=COLORS["accent"]),
        )
        .encode(
            x=alt.X("hour:O", axis=_themed_axis("Hour of day")),
            y=alt.Y("value:Q", axis=_themed_axis(metric_label), scale=alt.Scale(zero=False, padding=10)),
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("value:Q", title=metric_label, format=".2f"),
            ],
        )
    )
    rule_mean = alt.Chart(pd.DataFrame({"y": [mean_val]})).mark_rule(
        color=COLORS["muted"], strokeDash=[4, 4], strokeWidth=1.2,
    ).encode(y="y:Q")
    rule_ucl = alt.Chart(pd.DataFrame({"y": [ucl]})).mark_rule(
        color=COLORS["warning"], strokeDash=[2, 4], strokeWidth=1.2, opacity=0.85,
    ).encode(y="y:Q")
    rule_lcl = alt.Chart(pd.DataFrame({"y": [lcl]})).mark_rule(
        color=COLORS["warning"], strokeDash=[2, 4], strokeWidth=1.2, opacity=0.85,
    ).encode(y="y:Q")

    chart = (
        alt.layer(rule_ucl, rule_lcl, rule_mean, line)
        .properties(height=340, background="transparent")
        .configure_view(strokeOpacity=0)
    )

    st.markdown(
        f"<div class='snd-section' style='margin-top:18px'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Mean {mean_val:.2f} &middot; UCL {ucl:.2f} &middot; LCL {lcl:.2f} "
        f"(+/- 3 sigma)",
        unsafe_allow_html=True,
    )


def _render_correlation_native(
    primary: str, secondary: str, start_time: str, end_time: str, label: str
) -> None:
    """Two metrics on independent y-axes via Altair so each keeps its own scale."""
    from scipy.stats import pearsonr

    df = _query_hourly_metrics(start_time, end_time)
    if df.empty:
        st.info("No data in the selected window.")
        return

    if df[primary].std() > 0 and df[secondary].std() > 0 and len(df) >= 2:
        r, _ = pearsonr(df[primary], df[secondary])
        r_text = f"r = {r:+.3f}"
    else:
        r_text = "r = n/a"

    primary_color = COLORS["accent"]
    secondary_color = COLORS["cyan"]
    base = alt.Chart(df).encode(x=alt.X("short_hour:O", axis=_themed_axis("Hour of day")))
    line1 = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=45, color=primary_color),
        strokeWidth=2.5,
        color=primary_color,
    ).encode(
        y=alt.Y(
            f"{primary}:Q",
            axis=alt.Axis(
                title=METRIC_LABELS.get(primary, primary),
                titleColor=primary_color,
                labelColor=COLORS["muted"],
                gridColor=COLORS["grid"],
                domainColor=COLORS["border_strong"],
                tickColor=COLORS["border_strong"],
            ),
            scale=alt.Scale(zero=False, padding=10),
        ),
    )
    line2 = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=45, color=secondary_color),
        strokeWidth=2.5,
        color=secondary_color,
    ).encode(
        y=alt.Y(
            f"{secondary}:Q",
            axis=alt.Axis(
                title=METRIC_LABELS.get(secondary, secondary),
                titleColor=secondary_color,
                labelColor=COLORS["muted"],
                gridColor=COLORS["grid"],
                domainColor=COLORS["border_strong"],
                tickColor=COLORS["border_strong"],
            ),
            scale=alt.Scale(zero=False, padding=10),
        ),
    )
    chart = (
        alt.layer(line1, line2)
        .resolve_scale(y="independent")
        .properties(height=340, background="transparent")
        .configure_view(strokeOpacity=0)
    )

    st.markdown(
        f"<div class='snd-section' style='margin-top:18px'>{label} <span style='color:{COLORS['muted']}; text-transform:none; letter-spacing:0; font-weight:500;'>&middot; {r_text}</span></div>",
        unsafe_allow_html=True,
    )
    st.altair_chart(chart, use_container_width=True)


def _render_failure_timeline_native(start_time: str, end_time: str, label: str) -> None:
    """Scatter of failed chips, y-axis is failure_reason, color by reason."""
    sql = """
        SELECT timestamp, failure_reason
        FROM chip_production_data
        WHERE test_result = 'FAIL' AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """
    with _ro_connection() as conn:
        df = pd.read_sql_query(sql, conn, params=(start_time, end_time))

    st.markdown(
        f"<div class='snd-section' style='margin-top:18px'>{label}</div>",
        unsafe_allow_html=True,
    )
    if df.empty:
        st.info("No failures in the selected window.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["reason"] = df["failure_reason"].map(_friendly_failure_reason)

    # Reason -> palette mapping, keyed off the friendly labels.
    failure_palette = [
        COLORS["accent"],     # red
        COLORS["warning"],    # amber
        COLORS["cyan"],       # cyan
        COLORS["success"],    # green
        "#B57EDC",            # violet
    ]
    chart = (
        alt.Chart(df)
        .mark_circle(size=42, opacity=0.75, stroke=COLORS["bg"], strokeWidth=0.5)
        .encode(
            x=alt.X("timestamp:T", axis=_themed_axis("Timestamp")),
            y=alt.Y("reason:N", axis=_themed_axis("Failure reason")),
            color=alt.Color(
                "reason:N",
                scale=alt.Scale(range=failure_palette),
                legend=alt.Legend(
                    title="",
                    labelColor=COLORS["muted"],
                    symbolStrokeWidth=0,
                    orient="bottom",
                ),
            ),
            tooltip=["timestamp:T", "reason:N"],
        )
        # Explicit bottom padding so the "Timestamp" axis title is not
        # clipped by the Streamlit container at height=300.
        .properties(
            height=340,
            background="transparent",
            padding={"left": 5, "top": 5, "right": 5, "bottom": 45},
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_chart_call(call: dict[str, Any]) -> None:
    """Dispatch one generate_chart trace entry to the right native renderer."""
    args = call.get("input") or {}
    chart_type = args.get("chart_type")
    start_time = args.get("start_time")
    end_time = args.get("end_time")
    if not (chart_type and start_time and end_time):
        return

    base = CHART_TYPE_LABELS.get(chart_type, chart_type)
    if chart_type == "spc_chart":
        metric = args.get("primary_metric")
        if metric:
            _render_spc_native(
                metric, start_time, end_time,
                f"{base}: {METRIC_LABELS.get(metric, metric)}",
            )
    elif chart_type == "correlation_chart":
        primary = args.get("primary_metric")
        secondary = args.get("secondary_metric")
        if primary and secondary:
            _render_correlation_native(
                primary, secondary, start_time, end_time,
                f"{METRIC_LABELS.get(primary, primary)} vs {METRIC_LABELS.get(secondary, secondary)}",
            )
    elif chart_type == "failure_timeline":
        _render_failure_timeline_native(start_time, end_time, base)


CHART_TOKEN_RE = re.compile(r"\{\{\s*chart\s*:\s*([^}\n]+?)\s*\}\}")


def _find_chart_call_for_spec(
    spec: str, chart_calls: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Match a {{chart:...}} spec to a generate_chart call.

    A spec like 'spc_chart:npu_tops' matches a call whose chart_type
    and primary_metric agree. Extra parts (secondary_metric) tighten
    the match further. Empty spec components are ignored.
    """
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    if not parts:
        return None
    chart_type = parts[0]
    primary = parts[1] if len(parts) > 1 else None
    secondary = parts[2] if len(parts) > 2 else None

    for call in chart_calls:
        args = call.get("input") or {}
        if args.get("chart_type") != chart_type:
            continue
        if primary is not None and args.get("primary_metric") != primary:
            continue
        if secondary is not None and args.get("secondary_metric") != secondary:
            continue
        return call
    return None


def _render_markdown_with_charts(
    text: str | None, chart_calls: list[dict[str, Any]]
) -> set[int]:
    """Render markdown text, replacing {{chart:...}} tokens with inline charts.

    Returns the set of id()s of chart calls that were rendered, so the
    caller can detect any unreferenced charts and fall them back to
    end-of-page rendering.
    """
    rendered_ids: set[int] = set()
    if not text:
        return rendered_ids

    last_end = 0
    for match in CHART_TOKEN_RE.finditer(text):
        if match.start() > last_end:
            chunk = text[last_end:match.start()].strip("\n")
            if chunk:
                st.markdown(chunk)

        spec = match.group(1)
        call = _find_chart_call_for_spec(spec, chart_calls)
        if call is not None:
            _render_chart_call(call)
            rendered_ids.add(id(call))

        last_end = match.end()

    if last_end < len(text):
        rest = text[last_end:].strip("\n")
        if rest:
            st.markdown(rest)

    return rendered_ids


# ---------------------------------------------------------------------------
# Multi-turn chat: state keys and rendering
# ---------------------------------------------------------------------------
#
# The agent loop is wrapped in a Claude-style chat: a running history above,
# a chat_input pinned at the bottom, Stop + Clear chat buttons injected into
# the input's DOM. Follow-up questions inherit the full prior conversation,
# so the user can ask "go deeper on that finding" or "what about W050?"
# without re-specifying context.

MESSAGES_KEY = "chat_messages"   # full Anthropic message history (api-shaped)
TURNS_KEY = "chat_turns"         # list of UI turn records, one per Q/A pair
PENDING_KEY = "pending_question" # set on submit, drained on next rerun
RUNNING_KEY = "agent_running"    # True while the streaming loop is executing
PREFILL_KEY = "chat_prefill"     # staged text for the JS chat_input prefill


def _render_query_label(question: str) -> None:
    """The 'Query ->' label above a turn's content."""
    st.markdown(
        f'<div class="snd-query-label">'
        f'<span class="snd-query-arrow">Query &rarr;</span>'
        f'<span class="snd-query-text">{question}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_answer_body(turn: dict[str, Any]) -> None:
    """Render the structured report with inline charts, or fall back to prose."""
    report = turn.get("report_markdown")
    chart_calls = turn.get("chart_calls") or []
    rendered: set[int] = set()
    if report:
        rendered = _render_markdown_with_charts(report, chart_calls)
    elif turn.get("answer"):
        rendered = _render_markdown_with_charts(turn["answer"], chart_calls)
    # Drop unreferenced charts at the end so they are not lost.
    for call in chart_calls:
        if id(call) not in rendered:
            _render_chart_call(call)


def _render_past_turn(turn: dict[str, Any]) -> None:
    """Static replay of a past turn: divider, query, collapsed trace, answer."""
    st.divider()
    _render_query_label(turn["question"])
    trace_events = turn.get("trace_events") or []
    if trace_events:
        with st.expander("Thinking Breakdown", expanded=False):
            for kind, payload in trace_events:
                if kind == "text":
                    st.markdown(payload)
                else:  # tool_call
                    _render_trace_line(payload)
    if turn.get("error"):
        st.error(turn["error"])
    _render_answer_body(turn)
    iterations = turn.get("iterations") or 0
    total_ms = turn.get("total_ms") or 0
    if iterations:
        st.caption(
            f"Ran {iterations} iteration{'s' if iterations != 1 else ''} "
            f"in {total_ms / 1000:.1f} s."
        )


def render_chat_history() -> None:
    """Render every past turn in chronological order."""
    for turn in st.session_state.get(TURNS_KEY, []):
        _render_past_turn(turn)


def _run_new_turn(question: str) -> None:
    """Stream the agent for one new turn, render live, then save to history.

    The full prior message history is sent to the agent so follow-up
    questions ("dig deeper", "what about W050?") inherit the prior tool
    calls and assistant turns.
    """
    prior_messages = st.session_state.get(MESSAGES_KEY, [])
    messages = prior_messages + [{"role": "user", "content": question}]

    st.divider()
    _render_query_label(question)

    status_slot = st.empty()
    status_slot.markdown(
        '<div class="snd-thinking">'
        '<span class="snd-thinking-dots"><span></span><span></span><span></span></span>'
        'Thinking</div>',
        unsafe_allow_html=True,
    )

    trace_events: list[tuple[str, Any]] = []
    chart_calls: list[dict[str, Any]] = []
    final_messages = messages
    answer = ""
    report_markdown: str | None = None
    iterations = 0
    total_ms = 0.0
    error_message: str | None = None

    log_slot = st.empty()
    with log_slot.container():
        with st.expander("Thinking Breakdown", expanded=True):
            log_container = st.container()
            try:
                for event in run_agent_stream(messages):
                    etype = event["type"]
                    if etype == "assistant_text":
                        text = event["text"]
                        with log_container:
                            st.markdown(text)
                        trace_events.append(("text", text))
                    elif etype == "tool_call":
                        call = event["call"]
                        if call.get("name") == "generate_chart":
                            chart_calls.append(call)
                        if call.get("report"):
                            report_markdown = call["report"]
                        with log_container:
                            _render_trace_line(call)
                        trace_events.append(("tool_call", call))
                    elif etype == "final":
                        result = event["result"]
                        answer = result.answer
                        final_messages = result.messages
                        iterations = result.iterations
                        total_ms = result.total_duration_ms
            except Exception as exc:  # noqa: BLE001 (surface anything cleanly)
                err = str(exc)
                if "rate_limit" in err.lower() or "429" in err:
                    error_message = (
                        "Anthropic rate limit reached (30,000 input tokens "
                        "per minute on the default tier). Wait roughly "
                        "60 seconds and try again."
                    )
                else:
                    error_message = f"{type(exc).__name__}: {exc}"

    status_slot.empty()

    # Collapse the expander with replayed content so the answer becomes
    # the focus once streaming finishes.
    log_slot.empty()
    if trace_events:
        with log_slot.container():
            with st.expander("Thinking Breakdown", expanded=False):
                for kind, payload in trace_events:
                    if kind == "text":
                        st.markdown(payload)
                    else:
                        _render_trace_line(payload)

    turn_record = {
        "question": question,
        "trace_events": trace_events,
        "answer": answer,
        "report_markdown": report_markdown,
        "chart_calls": chart_calls,
        "iterations": iterations,
        "total_ms": total_ms,
        "error": error_message,
    }

    if error_message:
        st.error(error_message)

    _render_answer_body(turn_record)

    if iterations:
        st.caption(
            f"Ran {iterations} iteration{'s' if iterations != 1 else ''} "
            f"in {total_ms / 1000:.1f} s."
        )

    # Persist so the turn survives reruns and the next submission inherits
    # this turn's context.
    st.session_state[MESSAGES_KEY] = final_messages
    st.session_state.setdefault(TURNS_KEY, []).append(turn_record)


def _prefill_chat(q: str) -> None:
    """Sample-chip callback: stage `q` for JS prefill into the chat_input."""
    st.session_state[PREFILL_KEY] = q


def _inject_chat_action_buttons(*, is_running: bool, has_history: bool) -> None:
    """Inject visible Stop + Clear-chat twins into the chat_input DOM.

    Streamlit's chat_input has no slots, so the only way to put buttons
    inside it is to (a) render hidden Streamlit buttons whose clicks
    trigger reruns, then (b) inject visible JS twins inside the
    chat_input's DOM that programmatically click the hidden buttons.

    The script polls briefly because Streamlit's React tree may not have
    mounted the hidden buttons or the chat_input by the time the iframe
    runs its first pass. Each pass is idempotent.
    """
    stop_flag = "true" if is_running else "false"
    clear_flag = "true" if has_history else "false"
    st.components.v1.html(
        f"""
        <script>
          (function() {{
            const doc = window.parent.document;
            const isRunning = {stop_flag};
            const showClear = {clear_flag};

            function findHiddenStop() {{
              for (const b of doc.querySelectorAll('button')) {{
                if (b.classList.contains('snd-stop-btn')) continue;
                if (b.classList.contains('snd-clear-btn')) continue;
                if ((b.textContent || '').trim() === '⏹ Stop') return b;
              }}
              return null;
            }}

            function findHiddenClear() {{
              for (const b of doc.querySelectorAll('button')) {{
                if (b.classList.contains('snd-stop-btn')) continue;
                if (b.classList.contains('snd-clear-btn')) continue;
                const t = (b.textContent || '').replace(/\\s+/g, ' ').trim();
                if (t === 'Clear chat') return b;
              }}
              return null;
            }}

            function hideEl(el) {{
              if (!el) return;
              el.style.position = 'absolute';
              el.style.left = '-10000px';
              el.style.width = '1px';
              el.style.height = '1px';
              el.style.overflow = 'hidden';
              el.style.margin = '0';
              el.style.padding = '0';
            }}

            function findContainer(start) {{
              let cur = start;
              for (let i = 0; i < 8; i++) {{
                if (!cur || !cur.parentElement) break;
                cur = cur.parentElement;
                const tid = cur.getAttribute && cur.getAttribute('data-testid');
                if (tid === 'stElementContainer') return cur;
                if (cur.classList && cur.classList.contains('element-container')) return cur;
              }}
              return null;
            }}

            function sendBtnAnchor(chatInput) {{
              // Last non-twin button in the chat_input -- in vanilla Streamlit
              // that is the Send button. Last (not first) so any future
              // add-ons that prepend buttons (mic, audio) don't break us.
              const all = chatInput.querySelectorAll(
                'button:not(.snd-stop-btn):not(.snd-clear-btn)'
              );
              return all.length ? all[all.length - 1] : null;
            }}

            function attachStop() {{
              const existing = doc.querySelector('.snd-stop-btn');
              if (!isRunning) {{
                if (existing) existing.remove();
                return true;
              }}
              const hidden = findHiddenStop();
              if (!hidden) return false;  // retry, hidden button not mounted yet
              hideEl(findContainer(doc.querySelector('.snd-stop-anchor')));
              hideEl(findContainer(hidden));
              // Always remove + re-create so the click handler's closure
              // belongs to THIS iframe. Otherwise after a rerun the twin
              // can outlive its source iframe and stop firing.
              if (existing) existing.remove();
              const chatInput = doc.querySelector('[data-testid="stChatInput"]');
              if (!chatInput) return false;
              const sendBtn = sendBtnAnchor(chatInput);
              if (!sendBtn) return false;
              const vis = doc.createElement('button');
              vis.className = 'snd-stop-btn';
              vis.type = 'button';
              vis.innerHTML = '&#9632;';
              vis.title = 'Stop';
              vis.style.cssText =
                'background:rgba(50,83,220,0.18);color:#3253DC;'
                + 'border:1px solid rgba(50,83,220,0.5);border-radius:50%;'
                + 'width:30px;height:30px;margin-right:6px;cursor:pointer;'
                + 'display:inline-flex;align-items:center;justify-content:center;'
                + 'font-size:11px;padding:0;flex-shrink:0;line-height:1;';
              vis.addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                const fresh = findHiddenStop();
                if (fresh) fresh.click();
              }});
              sendBtn.parentElement.insertBefore(vis, sendBtn);
              return true;
            }}

            function attachClear() {{
              const existing = doc.querySelector('.snd-clear-btn');
              if (!showClear) {{
                if (existing) existing.remove();
                return true;
              }}
              const hidden = findHiddenClear();
              if (!hidden) return false;  // retry, hidden button not mounted yet
              hideEl(findContainer(doc.querySelector('.snd-clear-anchor')));
              hideEl(findContainer(hidden));
              // Always remove + re-create. See attachStop for the why.
              if (existing) existing.remove();
              const chatInput = doc.querySelector('[data-testid="stChatInput"]');
              if (!chatInput) return false;
              const stopVis = chatInput.querySelector('.snd-stop-btn');
              const sendBtn = sendBtnAnchor(chatInput);
              const anchor = stopVis || sendBtn;
              if (!anchor) return false;
              const vis = doc.createElement('button');
              vis.className = 'snd-clear-btn';
              vis.type = 'button';
              vis.textContent = 'Clear chat';
              vis.title = 'Clear chat history';
              vis.style.cssText =
                'background:transparent;color:#5B6577;'
                + 'border:1px solid rgba(15,23,42,0.18);border-radius:14px;'
                + 'height:28px;padding:0 12px;margin-right:8px;cursor:pointer;'
                + 'display:inline-flex;align-items:center;justify-content:center;'
                + 'font-size:12px;font-weight:500;font-family:inherit;'
                + 'flex-shrink:0;line-height:1;white-space:nowrap;'
                + 'transition:all 0.15s;';
              vis.addEventListener('mouseenter', function() {{
                vis.style.color = '#1F3CA8';
                vis.style.borderColor = 'rgba(50,83,220,0.55)';
                vis.style.background = 'rgba(50,83,220,0.10)';
              }});
              vis.addEventListener('mouseleave', function() {{
                vis.style.color = '#5B6577';
                vis.style.borderColor = 'rgba(15,23,42,0.18)';
                vis.style.background = 'transparent';
              }});
              vis.addEventListener('click', function(e) {{
                e.preventDefault();
                e.stopPropagation();
                const fresh = findHiddenClear();
                if (fresh) fresh.click();
              }});
              anchor.parentElement.insertBefore(vis, anchor);
              return true;
            }}

            function tick() {{
              return attachStop() && attachClear();
            }}

            // First pass immediately, then poll every 80ms for up to ~1.6s
            // in case the Streamlit React tree mounts the buttons or
            // chat_input asynchronously after this iframe loads.
            if (tick()) return;
            let attempts = 0;
            const interval = setInterval(function() {{
              attempts++;
              if (tick() || attempts >= 20) clearInterval(interval);
            }}, 80);
          }})();
        </script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_inject_css()
_render_header()

_render_dashboard(_load_dashboard_data())

st.divider()
st.markdown(
    '<div class="snd-section">Investigate with the agent</div>',
    unsafe_allow_html=True,
)

has_history = bool(st.session_state.get(TURNS_KEY))

# Defensive: if running got stuck True without a pending question (Stop was
# clicked mid-stream and the streaming loop was interrupted), reset on this
# fresh rerun so the chat_input does not stay disabled.
if st.session_state.get(RUNNING_KEY) and not st.session_state.get(PENDING_KEY):
    st.session_state[RUNNING_KEY] = False

# Hidden Clear chat button. The visible twin gets injected into the
# chat_input by _inject_chat_action_buttons. Only render when there is
# history to clear, so the JS knows to remove the twin when appropriate.
if has_history:
    st.markdown('<span class="snd-clear-anchor"></span>', unsafe_allow_html=True)
    if st.button("Clear chat", key="clear_chat_btn"):
        for key in (MESSAGES_KEY, TURNS_KEY, PENDING_KEY):
            st.session_state.pop(key, None)
        st.rerun()

render_chat_history()

# Sample chips only on a fresh thread, so they don't crowd a conversation
# that's already in progress.
if not has_history:
    st.markdown(
        '<div class="snd-section" style="margin-top:6px; font-size:0.72rem;">Try a sample question</div>',
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(SAMPLE_QUESTIONS), 2):
        row = SAMPLE_QUESTIONS[row_start:row_start + 2]
        cols = st.columns(2)
        for col, sample in zip(cols, row):
            col.button(
                sample,
                on_click=_prefill_chat,
                args=[sample],
                key=f"sample_{sample}",
            )

# If a chip click on the previous rerun staged a prefill, inject JS to
# populate the bottom-pinned chat_input. st.chat_input has no programmatic
# `value`, so we set the textarea's React-tracked value via the native
# property setter and dispatch an input event for React to pick it up.
_prefill = st.session_state.pop(PREFILL_KEY, None)
if _prefill:
    _safe = json.dumps(_prefill)
    st.components.v1.html(
        f"""
        <script>
          (function() {{
            const doc = window.parent.document;
            const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
            if (!ta) return;
            const setter = Object.getOwnPropertyDescriptor(
                window.parent.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(ta, {_safe});
            ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
            ta.focus();
          }})();
        </script>
        """,
        height=0,
    )

# Stage submissions on a separate rerun so RUNNING_KEY flips True before
# the Stop button row renders and _run_new_turn starts blocking.
_question = st.session_state.pop(PENDING_KEY, None)
if _question:
    st.session_state[RUNNING_KEY] = True

is_running = bool(st.session_state.get(RUNNING_KEY))

# Hidden Stop button while the agent is streaming. The visible twin gets
# injected into the chat_input. Clicking the twin clicks this hidden
# button, which queues a Streamlit rerun, interrupting the streaming loop.
if is_running:
    st.markdown('<span class="snd-stop-anchor"></span>', unsafe_allow_html=True)
    st.button("⏹ Stop", key="stop_btn")

# Run the inject script every rerun so the twins stay in sync with state.
_inject_chat_action_buttons(is_running=is_running, has_history=has_history)

submitted = st.chat_input(
    "Agent is thinking..." if is_running else "Ask the agent...",
    key="chat_input",
    disabled=is_running,
)

if submitted and submitted.strip():
    st.session_state[PENDING_KEY] = submitted.strip()
    st.rerun()

if _question and _question.strip():
    try:
        _run_new_turn(_question.strip())
    finally:
        st.session_state[RUNNING_KEY] = False
    st.rerun()
