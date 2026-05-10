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

from agent.agent import TraceStep, run_agent  # noqa: E402
from agent.logging_setup import setup_file_logging  # noqa: E402

# One log file per Streamlit process; idempotent so reruns reuse it.
setup_file_logging()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Snapdragon Yield Analytics",
    layout="wide",
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


def _stream_step(step: TraceStep) -> None:
    """Write one TraceStep into the current Streamlit context.

    Called as the run_agent on_step callback. Inside `with st.status(...)`
    everything written here lands inside the status box. The output is
    user-facing prose, not raw function calls.
    """
    for text in step.text_blocks:
        text = text.strip()
        if text:
            st.write(text)
    for call in step.tool_calls:
        st.markdown(f"- {_friendly_tool_label(call)}")


def _collect_charts(result) -> list[tuple[str, str]]:
    """Return [(label, absolute_path), ...] for every generate_chart call."""
    out = []
    for step in result.trace:
        for call in step.tool_calls:
            path = call.get("chart_path")
            if path and Path(path).exists():
                label = call.get("input", {}).get("chart_type", call["name"])
                metric_bits = []
                pm = call.get("input", {}).get("primary_metric")
                sm = call.get("input", {}).get("secondary_metric")
                if pm:
                    metric_bits.append(pm)
                if sm:
                    metric_bits.append(f"vs {sm}")
                if metric_bits:
                    label = f"{label}: {' '.join(metric_bits)}"
                out.append((label, path))
    return out


def _collect_reports(result) -> list[str]:
    """Return the markdown body of every write_summary_report call."""
    out = []
    for step in result.trace:
        for call in step.tool_calls:
            report = call.get("report")
            if report:
                out.append(report)
    return out


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

    if today_yield < 0.85:
        status_label = "ANOMALOUS"
    elif today_yield < 0.93:
        status_label = "CAUTION"
    else:
        status_label = "HEALTHY"

    fb = today.get("failure_breakdown") or []
    top_failure = _friendly_failure_reason(fb[0]["failure_reason"] if fb else None)

    st.subheader(f"Today's production  ({data['today_date']})")

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 3])
    c1.metric("Chips tested", f"{today['total_chips']:,}")
    c2.metric("Yield", f"{today_yield:.1%}", delta=f"{delta_pp:+.1f} pp")
    c3.metric("Failures", f"{today['failed']:,}")
    c4.metric("Top failure", top_failure)
    c5.metric("Status", status_label)

    n_prior = len(data["per_day"]) - 1
    if n_prior > 0:
        st.caption(
            f"Today's yield {today_yield:.1%} vs prior {n_prior}-day average "
            f"{prior_yield:.1%} ({delta_pp:+.1f} percentage points)."
        )

    if data["today_hourly"]:
        hourly_df = pd.DataFrame(data["today_hourly"])
        hourly_df["hour"] = hourly_df["hour"].str[-2:]
        hourly_df["yield_pct"] = hourly_df["yield"] * 100
        st.line_chart(
            hourly_df.set_index("hour")["yield_pct"],
            height=220,
            x_label="Hour of day",
            y_label="Yield (%)",
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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
    """SPC chart: hourly mean line plus mean / UCL / LCL reference lines."""
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
    chart_df = pd.DataFrame(
        {
            "Hour": df["short_hour"],
            metric_label: df[metric].astype(float),
            "Mean": [float(spc["mean"])] * len(df),
            "UCL (+3 sigma)": [float(spc["ucl"])] * len(df),
            "LCL (-3 sigma)": [float(spc["lcl"])] * len(df),
        }
    ).set_index("Hour")

    st.markdown(f"**{label}**")
    st.line_chart(chart_df, height=300)


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

    base = alt.Chart(df).encode(x=alt.X("short_hour:O", title="Hour of day"))
    line1 = base.mark_line(point=True, strokeWidth=2.5, color="#5b8def").encode(
        y=alt.Y(
            f"{primary}:Q",
            title=METRIC_LABELS.get(primary, primary),
            axis=alt.Axis(titleColor="#5b8def"),
        ),
    )
    line2 = base.mark_line(point=True, strokeWidth=2.5, color="#ff6b6b").encode(
        y=alt.Y(
            f"{secondary}:Q",
            title=METRIC_LABELS.get(secondary, secondary),
            axis=alt.Axis(titleColor="#ff6b6b"),
        ),
    )
    chart = alt.layer(line1, line2).resolve_scale(y="independent").properties(height=300)

    st.markdown(f"**{label}**  ({r_text})")
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

    st.markdown(f"**{label}**")
    if df.empty:
        st.info("No failures in the selected window.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["reason"] = df["failure_reason"].map(_friendly_failure_reason)
    chart = (
        alt.Chart(df)
        .mark_circle(size=24, opacity=0.55)
        .encode(
            x=alt.X("timestamp:T", title="Timestamp"),
            y=alt.Y("reason:N", title="Failure reason"),
            color=alt.Color("reason:N", legend=alt.Legend(title="")),
        )
        # Explicit bottom padding so the "Timestamp" axis title is not
        # clipped by the Streamlit container at height=300.
        .properties(height=300, padding={"left": 5, "top": 5, "right": 5, "bottom": 45})
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


def _iter_chart_calls(result):
    """Yield each generate_chart call from the trace."""
    for step in result.trace:
        for call in step.tool_calls:
            if call.get("name") == "generate_chart":
                yield call


def _set_question(q: str) -> None:
    """Sample-question callback. Populates the input box and clears any
    previous result so the layout is not stale."""
    st.session_state["question_input"] = q
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_question", None)


with st.sidebar:
    st.header("About")
    st.write(
        "AI agent for yield root cause analysis on Snapdragon chip "
        "production, focused on Hexagon NPU performance binning. The "
        "agent orchestrates five predefined tools to deliver an "
        "investigation report in under a minute. Portfolio "
        "demonstration of structured tool-using AI agents in "
        "manufacturing analytics. All data is synthetic."
    )

    st.divider()
    st.header("Sample questions")
    for sample in SAMPLE_QUESTIONS:
        st.button(
            sample,
            on_click=_set_question,
            args=[sample],
            use_container_width=True,
            key=f"sample_{sample}",
        )

    st.divider()
    st.header("How the agent works")
    st.write(
        "Each question is sent to a Claude agent that orchestrates five "
        "predefined tools, queries the production database, runs SPC "
        "and anomaly detection, generates charts, and writes a "
        "structured report."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Snapdragon Yield Analytics")
st.caption(
    "AI agent for Hexagon NPU yield investigation on Snapdragon SoCs."
)

_render_dashboard(_load_dashboard_data())

st.divider()
st.subheader("Investigate with the agent")

question = st.text_input(
    "Question",
    placeholder="for example: Why did yield drop this afternoon?",
    key="question_input",
)

run_col, clear_col, _ = st.columns([1, 1, 6])
with run_col:
    submit = st.button("Run", type="primary", disabled=not question.strip())
with clear_col:
    if "last_result" in st.session_state:
        if st.button("Clear"):
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_question", None)
            st.session_state.pop("pending_question", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

if submit and question.strip():
    q = question.strip()
    st.session_state["last_question"] = q
    st.session_state.pop("last_result", None)

    with st.status("Working...", expanded=True) as status:
        try:
            result = run_agent(q, on_step=_stream_step)
            status.update(
                label=(
                    f"Completed in {result.total_duration_ms / 1000:.1f} s "
                    f"across {result.iterations} iteration"
                    f"{'s' if result.iterations != 1 else ''}"
                ),
                state="complete",
            )
            st.session_state["last_result"] = result
        except Exception as exc:  # noqa: BLE001  (we want to surface anything)
            err_msg = str(exc)
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                status.update(
                    label="Rate limit reached. Wait about a minute and try again.",
                    state="error",
                )
                st.warning(
                    "Anthropic rate limit hit (30,000 input tokens per minute "
                    "on the default tier). Wait roughly 60 seconds and re-run, "
                    "or upgrade the API tier."
                )
            else:
                status.update(label=f"Error: {exc}", state="error")
                st.exception(exc)
            st.stop()


# ---------------------------------------------------------------------------
# Render the most recent result
# ---------------------------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.divider()

    # The agent's deliverable is the markdown report. Findings carry
    # {{chart:...}} tokens that the renderer expands into inline charts.
    # If the agent did not produce a report (for example, an ambiguous
    # question that did not warrant a structured investigation), we fall
    # back to its short prose answer so the user still sees something.
    chart_calls = list(_iter_chart_calls(result))
    rendered_chart_ids: set[int] = set()

    reports = _collect_reports(result)
    if reports:
        rendered_chart_ids |= _render_markdown_with_charts(reports[-1], chart_calls)
        if len(reports) > 1:
            with st.expander(f"Earlier report drafts ({len(reports) - 1})"):
                for i, r in enumerate(reports[:-1], 1):
                    st.markdown(f"**Draft {i}**")
                    _render_markdown_with_charts(r, chart_calls)
    else:
        rendered_chart_ids |= _render_markdown_with_charts(result.answer, chart_calls)

    # Fallback: any chart the agent generated but did not reference via
    # an inline token gets rendered at the end so it is not lost.
    unreferenced = [c for c in chart_calls if id(c) not in rendered_chart_ids]
    for call in unreferenced:
        _render_chart_call(call)

    with st.expander("Full agent trace", expanded=False):
        for step in result.trace:
            st.markdown(
                f"**Step {step.iteration}** | {step.duration_ms:.0f} ms | "
                f"stop_reason = `{step.stop_reason}`"
            )
            for text in step.text_blocks:
                text = text.strip()
                if text:
                    st.markdown(f"> {text}")
            for call in step.tool_calls:
                args = _pretty_args(call.get("input"))
                st.code(
                    f"{call['name']}({args})\n  -> {call['result_summary']}",
                    language="text",
                )
            st.markdown("")

    st.caption(
        f"Agent ran {result.iterations} iteration"
        f"{'s' if result.iterations != 1 else ''} in "
        f"{result.total_duration_ms:.0f} ms total."
    )
