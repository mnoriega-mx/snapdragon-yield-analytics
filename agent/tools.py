"""
Tool catalog for the Snapdragon Yield Analytics agent.

This file is the single source of truth for the tools Claude can call.
Each tool has two halves:

    1. A Python implementation (a regular function) that does the work
       against the SQLite database or in-memory data.
    2. An Anthropic tool schema (a dictionary in the JSON Schema format
       the Messages API expects) that describes the tool to Claude.

`TOOL_SCHEMAS` is the list passed to the Messages API as `tools=...`.
`TOOL_IMPLEMENTATIONS` maps a tool name to the Python function that
runs when Claude asks for it.

Day 2 ships only Tool 1 (query_database). Tools 2 to 5 will be appended
to the same dictionaries on Days 3 to 5.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")  # render to file without needing a display

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = (Path(__file__).resolve().parent.parent / "data" / "chip_production.db")

# Cap row counts returned to the agent so a careless query cannot blow up
# the context window. 10k rows of test data is the entire daily lot.
MAX_ROWS_RETURNED = 2_000


@contextmanager
def _connect(db_path: Path):
    """Open a read-only SQLite connection.

    Using a context manager makes sure the connection is always closed,
    even if the query raises. Read-only mode (`mode=ro`) is a defensive
    layer: the agent should never be able to mutate the data, even by
    accident.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        # Return rows as plain dictionaries so JSON serialization is trivial.
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 1: query_database
# ---------------------------------------------------------------------------

VALID_QUERY_TYPES = {"date_range", "wafer_range", "failed_only", "summary"}


def query_database(
    query_type: str,
    start_time: str | None = None,
    end_time: str | None = None,
    wafer_ids: Iterable[str] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Run a parameterized read against the chip production database.

    The agent never gets free-form SQL; it picks one of four query types
    and supplies optional filters. This matches the FSDO job description's
    "predefined code paths" requirement and is much safer than letting an
    LLM author SQL.

    Args:
        query_type: One of 'date_range', 'wafer_range', 'failed_only',
            'summary'.
        start_time: ISO 8601 timestamp (inclusive). Required for
            date_range and failed_only when constraining time. Optional
            for summary.
        end_time: ISO 8601 timestamp (exclusive of the next second).
            Same applies as start_time.
        wafer_ids: Iterable of wafer identifiers (for example
            ['W050', 'W051']). Required for wafer_range.
        db_path: Path to the SQLite database. Defaults to
            data/chip_production.db relative to the project root.

    Returns:
        A dictionary with three keys:
            'query_type'   : the resolved query type
            'row_count'    : number of rows the underlying query matched
            'returned_rows': up to MAX_ROWS_RETURNED rows as JSON-friendly dicts
            'truncated'    : True if row_count exceeded MAX_ROWS_RETURNED
            'summary'      : present only for query_type='summary'

    Raises:
        ValueError: when arguments are missing or query_type is invalid.
        FileNotFoundError: when the database file does not exist.
    """
    if query_type not in VALID_QUERY_TYPES:
        raise ValueError(
            f"query_type must be one of {sorted(VALID_QUERY_TYPES)}, got {query_type!r}"
        )

    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.exists():
        raise FileNotFoundError(
            f"Database not found at {db}. Run "
            "`python data/generate_data.py && python data/setup_database.py` first."
        )

    with _connect(db) as conn:
        if query_type == "summary":
            return _summary(conn, start_time, end_time)

        sql, params = _build_select(query_type, start_time, end_time, wafer_ids)
        rows = conn.execute(sql, params).fetchall()
        row_count = len(rows)
        returned = [dict(r) for r in rows[:MAX_ROWS_RETURNED]]
        return {
            "query_type": query_type,
            "row_count": row_count,
            "returned_rows": returned,
            "truncated": row_count > MAX_ROWS_RETURNED,
        }


def _build_select(
    query_type: str,
    start_time: str | None,
    end_time: str | None,
    wafer_ids: Iterable[str] | None,
) -> tuple[str, list[Any]]:
    """Compose the SELECT statement and parameter list for a query type.

    All values flow into the query as bound parameters, never as string
    interpolation, so the agent cannot smuggle in SQL.
    """
    base = (
        "SELECT timestamp, wafer_id, chip_id, soc_model, process_node, "
        "npu_tops, npu_power_w, cpu_freq_ghz, memory_bandwidth_gbps, "
        "die_temp_c, test_result, failure_reason "
        "FROM chip_production_data"
    )
    where: list[str] = []
    params: list[Any] = []

    if query_type == "date_range":
        if not start_time or not end_time:
            raise ValueError("date_range requires both start_time and end_time")
        where.append("timestamp >= ?")
        where.append("timestamp < ?")
        params.extend([start_time, end_time])

    elif query_type == "wafer_range":
        wafer_list = list(wafer_ids or [])
        if not wafer_list:
            raise ValueError("wafer_range requires at least one wafer_id")
        placeholders = ", ".join("?" for _ in wafer_list)
        where.append(f"wafer_id IN ({placeholders})")
        params.extend(wafer_list)
        if start_time:
            where.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where.append("timestamp < ?")
            params.append(end_time)

    elif query_type == "failed_only":
        where.append("test_result = 'FAIL'")
        if start_time:
            where.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where.append("timestamp < ?")
            params.append(end_time)

    sql = base
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp ASC"
    return sql, params


def _summary(conn: sqlite3.Connection, start_time: str | None, end_time: str | None) -> dict[str, Any]:
    """Return a compact aggregate summary, optionally bounded by time.

    This is the cheap query the agent should reach for first when the
    user asks something open-ended like 'how is yield today?'. It avoids
    pulling thousands of rows just to get a few aggregate numbers.
    """
    where = []
    params: list[Any] = []
    if start_time:
        where.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        where.append("timestamp < ?")
        params.append(end_time)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    totals = conn.execute(
        f"""
        SELECT
            COUNT(*) AS total_chips,
            SUM(CASE WHEN test_result = 'PASS' THEN 1 ELSE 0 END) AS passed,
            SUM(CASE WHEN test_result = 'FAIL' THEN 1 ELSE 0 END) AS failed,
            MIN(timestamp) AS first_timestamp,
            MAX(timestamp) AS last_timestamp
        FROM chip_production_data{where_sql}
        """,
        params,
    ).fetchone()

    total = int(totals["total_chips"] or 0)
    passed = int(totals["passed"] or 0)
    failed = int(totals["failed"] or 0)
    yield_pct = (passed / total) if total else 0.0

    failure_breakdown = conn.execute(
        f"""
        SELECT failure_reason, COUNT(*) AS n
        FROM chip_production_data
        {where_sql + (' AND ' if where_sql else ' WHERE ')}test_result = 'FAIL'
        GROUP BY failure_reason
        ORDER BY n DESC
        """,
        params,
    ).fetchall()

    hourly = conn.execute(
        f"""
        SELECT
            substr(timestamp, 1, 13) AS hour,
            COUNT(*) AS n,
            SUM(CASE WHEN test_result = 'PASS' THEN 1 ELSE 0 END) AS passed
        FROM chip_production_data{where_sql}
        GROUP BY hour
        ORDER BY hour
        """,
        params,
    ).fetchall()

    # Build a daily rollup so multi-day windows have a compact summary
    # without dragging 168 hourly entries into the agent's context.
    daily_map: dict[str, dict[str, int]] = {}
    for r in hourly:
        d = r["hour"][:10]
        slot = daily_map.setdefault(d, {"n": 0, "passed": 0})
        slot["n"] += int(r["n"])
        slot["passed"] += int(r["passed"])
    daily_yield = sorted(
        (
            {
                "date": d,
                "n": v["n"],
                "passed": v["passed"],
                "yield": round(v["passed"] / v["n"], 4) if v["n"] else 0.0,
            }
            for d, v in daily_map.items()
        ),
        key=lambda x: x["date"],
    )

    # Cap hourly_yield to the last day if the window spans multiple days,
    # so a no-window summary on a 7-day database stays small.
    if len(daily_yield) > 1:
        last_day = daily_yield[-1]["date"]
        hourly_in_scope = [r for r in hourly if r["hour"][:10] == last_day]
    else:
        hourly_in_scope = list(hourly)

    return {
        "query_type": "summary",
        "row_count": total,
        "returned_rows": [],
        "truncated": False,
        "summary": {
            "total_chips": total,
            "passed": passed,
            "failed": failed,
            "yield": round(yield_pct, 4),
            "first_timestamp": totals["first_timestamp"],
            "last_timestamp": totals["last_timestamp"],
            "failure_breakdown": [
                {"failure_reason": r["failure_reason"], "count": int(r["n"])}
                for r in failure_breakdown
            ],
            "hourly_yield": [
                {
                    "hour": r["hour"],
                    "n": int(r["n"]),
                    "passed": int(r["passed"]),
                    "yield": round(r["passed"] / r["n"], 4) if r["n"] else 0.0,
                }
                for r in hourly_in_scope
            ],
            "daily_yield": daily_yield,
        },
    }


# ---------------------------------------------------------------------------
# Shared helper: load a window of chips into a DataFrame
# ---------------------------------------------------------------------------

def _load_dataframe(
    conn: sqlite3.Connection,
    start_time: str | None,
    end_time: str | None,
) -> pd.DataFrame:
    """Read chips in the half-open window [start_time, end_time) into a DataFrame.

    Used by Tool 2 and Tool 3 so they can do their math in pandas. SQLite
    stores the timestamp as TEXT, so the column comes back as a string
    series ordered ascending.
    """
    sql = (
        "SELECT timestamp, wafer_id, chip_id, "
        "npu_tops, npu_power_w, cpu_freq_ghz, memory_bandwidth_gbps, "
        "die_temp_c, test_result, failure_reason "
        "FROM chip_production_data"
    )
    where: list[str] = []
    params: list[Any] = []
    if start_time:
        where.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        where.append("timestamp < ?")
        params.append(end_time)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp ASC"
    return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Tool 2: calculate_spc_metrics
# ---------------------------------------------------------------------------

VALID_METRICS = {
    "npu_tops",
    "npu_power_w",
    "cpu_freq_ghz",
    "memory_bandwidth_gbps",
    "die_temp_c",
}
VALID_GROUP_BY = {"hour", "wafer_id"}


def calculate_spc_metrics(
    metric: str,
    start_time: str,
    end_time: str,
    group_by: str,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Compute SPC mean, std, and 3-sigma control limits for one metric.

    Mean and sample standard deviation are computed on the chip-level
    values inside the window. Limits are set at UCL = mean + 3*std and
    LCL = mean - 3*std (the brief's "mean +/- 3 sigma"). The data is
    then aggregated into subgroups (by calendar hour or by wafer_id),
    and any subgroup whose mean falls outside [LCL, UCL] is reported in
    `out_of_control`.

    Args:
        metric: Column name. One of npu_tops, npu_power_w, cpu_freq_ghz,
            memory_bandwidth_gbps, die_temp_c.
        start_time: Inclusive ISO 8601 timestamp lower bound.
        end_time: Exclusive ISO 8601 timestamp upper bound.
        group_by: 'hour' aggregates by calendar hour; 'wafer_id'
            aggregates by wafer.
        db_path: Optional override for the database path.

    Returns:
        Dict with metric, window, n, mean, std, ucl, lcl, the full list
        of subgroup means, and the subset of subgroups out of control.

    Raises:
        ValueError: invalid metric, invalid group_by, missing time
            bounds, or an empty window.
        FileNotFoundError: the database file does not exist.
    """
    if metric not in VALID_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(VALID_METRICS)}, got {metric!r}"
        )
    if group_by not in VALID_GROUP_BY:
        raise ValueError(
            f"group_by must be one of {sorted(VALID_GROUP_BY)}, got {group_by!r}"
        )
    if not start_time or not end_time:
        raise ValueError("calculate_spc_metrics requires both start_time and end_time")

    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.exists():
        raise FileNotFoundError(
            f"Database not found at {db}. Run "
            "`python data/generate_data.py && python data/setup_database.py` first."
        )

    with _connect(db) as conn:
        df = _load_dataframe(conn, start_time, end_time)

    if df.empty:
        raise ValueError(f"no rows in window [{start_time}, {end_time})")

    values = df[metric]
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    if group_by == "hour":
        df = df.assign(_group=df["timestamp"].astype(str).str[:13])
    else:
        df = df.assign(_group=df["wafer_id"])

    grouped = (
        df.groupby("_group", sort=True)[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"_group": "group"})
    )

    out_of_control_mask = (grouped["mean"] > ucl) | (grouped["mean"] < lcl)

    return {
        "metric": metric,
        "start_time": start_time,
        "end_time": end_time,
        "group_by": group_by,
        "n": int(len(values)),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ucl": round(ucl, 4),
        "lcl": round(lcl, 4),
        "groups": [
            {
                "group": str(row["group"]),
                "mean": round(float(row["mean"]), 4),
                "n": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ],
        "out_of_control": [
            {
                "group": str(row["group"]),
                "mean": round(float(row["mean"]), 4),
                "n": int(row["count"]),
            }
            for _, row in grouped[out_of_control_mask].iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Tool 3: detect_anomalies
# ---------------------------------------------------------------------------

CORRELATION_METRICS: tuple[str, ...] = (
    "npu_tops",
    "npu_power_w",
    "cpu_freq_ghz",
    "memory_bandwidth_gbps",
    "die_temp_c",
)


def detect_anomalies(
    start_time: str,
    end_time: str,
    failure_rate_threshold: float = 0.10,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Find anomalous hours and rank metrics by correlation with failures.

    Buckets chips into hourly windows, computes the per-hour failure
    rate, and flags hours whose rate is strictly greater than
    `failure_rate_threshold`. Then computes Pearson correlation between
    each metric's hourly mean and the hourly failure rate. A negative
    correlation on npu_tops paired with a positive one on npu_power_w
    fingerprints a power-domain excursion on the NPU.

    Args:
        start_time: Inclusive ISO 8601 lower bound.
        end_time: Exclusive ISO 8601 upper bound.
        failure_rate_threshold: Float in [0, 1]. Default 0.10 (10 percent).
        db_path: Optional database path override.

    Returns:
        Dict with overall stats, anomalous_windows, hourly_summary, and
        a correlations sub-dict keyed by metric name.

    Raises:
        ValueError: missing time bounds, threshold out of range, or
            empty window.
        FileNotFoundError: database file missing.
    """
    if not start_time or not end_time:
        raise ValueError("detect_anomalies requires both start_time and end_time")
    if not isinstance(failure_rate_threshold, (int, float)):
        raise ValueError("failure_rate_threshold must be a number")
    if not (0 <= failure_rate_threshold <= 1):
        raise ValueError("failure_rate_threshold must be between 0 and 1")

    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.exists():
        raise FileNotFoundError(
            f"Database not found at {db}. Run "
            "`python data/generate_data.py && python data/setup_database.py` first."
        )

    with _connect(db) as conn:
        df = _load_dataframe(conn, start_time, end_time)

    if df.empty:
        raise ValueError(f"no rows in window [{start_time}, {end_time})")

    df = df.assign(
        _hour=df["timestamp"].astype(str).str[:13],
        _failed=(df["test_result"] == "FAIL").astype(int),
    )

    agg_kwargs: dict[str, tuple[str, str]] = {
        "n": ("test_result", "count"),
        "failures": ("_failed", "sum"),
    }
    for m in CORRELATION_METRICS:
        agg_kwargs[f"{m}_mean"] = (m, "mean")

    hourly = (
        df.groupby("_hour", sort=True)
        .agg(**agg_kwargs)
        .reset_index()
        .rename(columns={"_hour": "hour"})
    )
    hourly["failure_rate"] = hourly["failures"] / hourly["n"]

    anomalous = hourly[hourly["failure_rate"] > failure_rate_threshold]

    correlations: dict[str, dict[str, float] | None] = {}
    if len(hourly) >= 2 and hourly["failure_rate"].std(ddof=1) > 0:
        for m in CORRELATION_METRICS:
            col = f"{m}_mean"
            if hourly[col].std(ddof=1) == 0:
                correlations[m] = None
                continue
            r, p = pearsonr(hourly[col], hourly["failure_rate"])
            correlations[m] = {
                "r": round(float(r), 4),
                "p": round(float(p), 6),
            }
    else:
        for m in CORRELATION_METRICS:
            correlations[m] = None

    overall_n = int(len(df))
    overall_failed = int(df["_failed"].sum())
    overall_rate = overall_failed / overall_n if overall_n else 0.0

    return {
        "start_time": start_time,
        "end_time": end_time,
        "threshold": float(failure_rate_threshold),
        "n_total": overall_n,
        "n_failed": overall_failed,
        "overall_failure_rate": round(overall_rate, 4),
        "anomalous_windows": [
            {
                "hour": row["hour"],
                "n": int(row["n"]),
                "failures": int(row["failures"]),
                "failure_rate": round(float(row["failure_rate"]), 4),
            }
            for _, row in anomalous.iterrows()
        ],
        "hourly_summary": [
            {
                "hour": row["hour"],
                "n": int(row["n"]),
                "failures": int(row["failures"]),
                "failure_rate": round(float(row["failure_rate"]), 4),
            }
            for _, row in hourly.iterrows()
        ],
        "correlations": correlations,
    }


# ---------------------------------------------------------------------------
# Tool 4: generate_chart
# ---------------------------------------------------------------------------

VALID_CHART_TYPES = {"spc_chart", "correlation_chart", "failure_timeline"}

DEFAULT_CHART_DIR = Path(__file__).resolve().parent.parent / "charts"

# Stable color per failure reason so the same reason always gets the same
# color across runs and across charts. Anything not listed falls back to gray.
FAILURE_REASON_COLORS: dict[str, str] = {
    "npu_tops_below_spec": "#E63946",
    "npu_power_above_spec": "#F4A261",
    "cpu_freq_below_spec": "#2A9D8F",
    "memory_bandwidth_low": "#264653",
    "die_temp_over_threshold": "#9D4EDD",
}

# Friendly y-axis labels for the metrics. Keeps charts self-explanatory.
METRIC_LABELS: dict[str, str] = {
    "npu_tops": "NPU TOPS",
    "npu_power_w": "NPU power (W)",
    "cpu_freq_ghz": "CPU frequency (GHz)",
    "memory_bandwidth_gbps": "Memory bandwidth (GB/s)",
    "die_temp_c": "Die temperature (C)",
}


def _short_hour(hour_string: str) -> str:
    """Render '2026-04-01 14' as '14' for compact x-axis labels."""
    return hour_string[-2:]


def _new_filename(prefix: str, suffix: str = "") -> str:
    """Build a unique PNG filename: prefix_YYYYMMDD_HHMMSS_microseconds.png."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    middle = f"_{suffix}" if suffix else ""
    return f"{prefix}{middle}_{stamp}.png"


def _draw_spc_chart(
    df: pd.DataFrame,
    metric: str,
    start_time: str,
    end_time: str,
    out_dir: Path,
) -> Path:
    """SPC chart: hourly mean line, mean and +/- 3 sigma reference lines, OOC in red."""
    df = df.assign(_hour=df["timestamp"].astype(str).str[:13])
    grouped = (
        df.groupby("_hour", sort=True)[metric]
        .agg(["mean", "count"])
        .reset_index()
    )

    values = df[metric]
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    ooc_mask = (grouped["mean"] > ucl) | (grouped["mean"] < lcl)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(grouped)))

    ax.plot(
        x, grouped["mean"],
        marker="o", linewidth=1.6, markersize=6,
        color="#2E86AB", label=f"hourly mean",
    )
    if ooc_mask.any():
        ax.scatter(
            [i for i, b in enumerate(ooc_mask) if b],
            grouped.loc[ooc_mask, "mean"],
            color="#E63946", s=110, zorder=5,
            label="out of control",
        )
    ax.axhline(mean, color="#2A9D8F", linestyle="--", alpha=0.8,
               label=f"mean = {mean:.2f}")
    ax.axhline(ucl, color="#E63946", linestyle=":", alpha=0.8,
               label=f"UCL = {ucl:.2f}")
    ax.axhline(lcl, color="#E63946", linestyle=":", alpha=0.8,
               label=f"LCL = {lcl:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_hour(h) for h in grouped["_hour"]])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"SPC chart: {METRIC_LABELS.get(metric, metric)}\n"
        f"{start_time} to {end_time} (synthetic data)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / _new_filename("spc", metric)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _draw_correlation_chart(
    df: pd.DataFrame,
    primary: str,
    secondary: str,
    start_time: str,
    end_time: str,
    out_dir: Path,
) -> Path:
    """Two metrics on dual y-axes, hour-bucketed; Pearson r in the title."""
    df = df.assign(_hour=df["timestamp"].astype(str).str[:13])
    grouped = (
        df.groupby("_hour", sort=True)
        .agg({primary: "mean", secondary: "mean"})
        .reset_index()
    )

    if (
        len(grouped) >= 2
        and grouped[primary].std(ddof=1) > 0
        and grouped[secondary].std(ddof=1) > 0
    ):
        r, _ = pearsonr(grouped[primary], grouped[secondary])
        r_text = f"r = {r:+.3f}"
    else:
        r_text = "r = n/a"

    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = list(range(len(grouped)))

    color1 = "#2E86AB"
    color2 = "#E63946"

    ax1.plot(x, grouped[primary], marker="o", color=color1, linewidth=1.6,
             label=METRIC_LABELS.get(primary, primary))
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel(METRIC_LABELS.get(primary, primary), color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_short_hour(h) for h in grouped["_hour"]])
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, grouped[secondary], marker="s", color=color2, linewidth=1.6,
             label=METRIC_LABELS.get(secondary, secondary))
    ax2.set_ylabel(METRIC_LABELS.get(secondary, secondary), color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(
        f"{METRIC_LABELS.get(primary, primary)} vs "
        f"{METRIC_LABELS.get(secondary, secondary)}  ({r_text})\n"
        f"{start_time} to {end_time} (synthetic data)"
    )

    # Combined legend across both axes.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=9)

    fig.tight_layout()

    path = out_dir / _new_filename("corr", f"{primary}_vs_{secondary}")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _draw_failure_timeline(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
    out_dir: Path,
) -> Path:
    """Scatter of every failed chip, y-axis is failure_reason, color by reason."""
    failures = df[df["test_result"] == "FAIL"].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    if failures.empty:
        ax.text(
            0.5, 0.5, "No failures in the selected window.",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        failures["timestamp_dt"] = pd.to_datetime(failures["timestamp"])
        reasons = sorted(failures["failure_reason"].dropna().unique())
        reason_to_y = {r: i for i, r in enumerate(reasons)}

        for reason in reasons:
            mask = failures["failure_reason"] == reason
            ax.scatter(
                failures.loc[mask, "timestamp_dt"],
                [reason_to_y[reason]] * int(mask.sum()),
                color=FAILURE_REASON_COLORS.get(reason, "#888888"),
                label=reason,
                alpha=0.55, s=22, edgecolors="none",
            )

        ax.set_yticks(list(reason_to_y.values()))
        ax.set_yticklabels(reasons)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Failure reason")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.3, axis="x")
        fig.autofmt_xdate()

    ax.set_title(
        f"Failure timeline\n{start_time} to {end_time} (synthetic data)"
    )
    fig.tight_layout()

    path = out_dir / _new_filename("failure_timeline")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_chart(
    chart_type: str,
    start_time: str,
    end_time: str,
    primary_metric: str | None = None,
    secondary_metric: str | None = None,
    output_dir: Path | str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Render one of three chart templates and return its file path.

    Args:
        chart_type: One of 'spc_chart', 'correlation_chart',
            'failure_timeline'.
        start_time: Inclusive ISO 8601 timestamp lower bound.
        end_time: Exclusive ISO 8601 timestamp upper bound.
        primary_metric: Required for spc_chart and correlation_chart;
            ignored for failure_timeline.
        secondary_metric: Required for correlation_chart; ignored
            otherwise. Must differ from primary_metric.
        output_dir: Where to save the PNG. Defaults to ./charts/.
        db_path: Optional override for the database path.

    Returns:
        A dict with chart_type, start_time, end_time, primary_metric,
        secondary_metric, the absolute file path, and the bare filename.

    Raises:
        ValueError: invalid chart_type, missing time bounds, missing or
            invalid metric for the chosen chart type, or empty window.
        FileNotFoundError: the database file does not exist.
    """
    if chart_type not in VALID_CHART_TYPES:
        raise ValueError(
            f"chart_type must be one of {sorted(VALID_CHART_TYPES)}, "
            f"got {chart_type!r}"
        )
    if not start_time or not end_time:
        raise ValueError("generate_chart requires both start_time and end_time")

    if chart_type == "spc_chart":
        if not primary_metric:
            raise ValueError("spc_chart requires primary_metric")
        if primary_metric not in VALID_METRICS:
            raise ValueError(
                f"primary_metric must be one of {sorted(VALID_METRICS)}, "
                f"got {primary_metric!r}"
            )
    elif chart_type == "correlation_chart":
        if not primary_metric or not secondary_metric:
            raise ValueError(
                "correlation_chart requires both primary_metric and secondary_metric"
            )
        if primary_metric not in VALID_METRICS:
            raise ValueError(
                f"primary_metric must be one of {sorted(VALID_METRICS)}, "
                f"got {primary_metric!r}"
            )
        if secondary_metric not in VALID_METRICS:
            raise ValueError(
                f"secondary_metric must be one of {sorted(VALID_METRICS)}, "
                f"got {secondary_metric!r}"
            )
        if primary_metric == secondary_metric:
            raise ValueError(
                "primary_metric and secondary_metric must differ for correlation_chart"
            )
    # failure_timeline: nothing else to validate.

    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.exists():
        raise FileNotFoundError(
            f"Database not found at {db}. Run "
            "`python data/generate_data.py && python data/setup_database.py` first."
        )

    out_dir = Path(output_dir) if output_dir else DEFAULT_CHART_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with _connect(db) as conn:
        df = _load_dataframe(conn, start_time, end_time)

    if df.empty:
        raise ValueError(f"no rows in window [{start_time}, {end_time})")

    if chart_type == "spc_chart":
        path = _draw_spc_chart(df, primary_metric, start_time, end_time, out_dir)
    elif chart_type == "correlation_chart":
        path = _draw_correlation_chart(
            df, primary_metric, secondary_metric, start_time, end_time, out_dir,
        )
    else:  # failure_timeline
        path = _draw_failure_timeline(df, start_time, end_time, out_dir)

    return {
        "chart_type": chart_type,
        "start_time": start_time,
        "end_time": end_time,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "path": str(path.resolve()),
        "filename": path.name,
    }


# ---------------------------------------------------------------------------
# Tool 5: write_summary_report
# ---------------------------------------------------------------------------

REQUIRED_FINDING_FIELDS = ("category", "description", "evidence")


def write_summary_report(
    findings: list[dict[str, str]],
    root_cause_hypothesis: str,
    recommendations: list[str],
) -> dict[str, Any]:
    """Render a markdown yield analysis report.

    The agent calls this as its final synthesis step. Findings are
    rendered in order as a flowing narrative, then a root-cause
    hypothesis that explains what the evidence points to, then a
    numbered list of recommendations.

    Args:
        findings: List of `{category, description, evidence}` dicts.
            All three fields are required strings. The description may
            contain `{{chart:...}}` tokens that the UI expands into an
            inline chart at that position.
        root_cause_hypothesis: One short paragraph (two to four
            sentences) explaining what the evidence converges on and
            naming the affected subsystem (NPU, CPU, memory, thermal).
        recommendations: List of plain-text recommendations.

    Returns:
        Dict with the rendered markdown plus simple counters.

    Raises:
        ValueError: structural issues with the inputs.
    """
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            raise ValueError(f"findings[{i}] must be an object")
        for key in REQUIRED_FINDING_FIELDS:
            if key not in f:
                raise ValueError(f"findings[{i}] is missing required field {key!r}")
            if not isinstance(f[key], str):
                raise ValueError(f"findings[{i}].{key} must be a string")

    if not isinstance(root_cause_hypothesis, str) or not root_cause_hypothesis.strip():
        raise ValueError("root_cause_hypothesis must be a non-empty string")

    if not isinstance(recommendations, list):
        raise ValueError("recommendations must be a list")
    for i, rec in enumerate(recommendations):
        if not isinstance(rec, str):
            raise ValueError(f"recommendations[{i}] must be a string")

    lines: list[str] = []
    lines.append(
        "_Generated "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} "
        "from synthetic Snapdragon production data._"
    )
    lines.append("")

    if findings:
        lines.append("## Findings")
        lines.append("")
        for f in findings:
            lines.append(f"### {f['category']}")
            lines.append("")
            lines.append(f["description"].strip())
            lines.append("")
            lines.append(f"_Evidence: {f['evidence'].strip()}_")
            lines.append("")

    lines.append("## Root cause hypothesis")
    lines.append("")
    lines.append(root_cause_hypothesis.strip())
    lines.append("")

    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec.strip()}")
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"

    return {
        "report": report,
        "n_findings": len(findings),
        "n_recommendations": len(recommendations),
        "char_count": len(report),
    }


# ---------------------------------------------------------------------------
# Anthropic tool schemas
# ---------------------------------------------------------------------------

QUERY_DATABASE_SCHEMA: dict[str, Any] = {
    "name": "query_database",
    "description": (
        "Run a parameterized query against the Snapdragon production test "
        "database. Use this whenever you need to read raw data or get a "
        "quick aggregate summary. The query_type 'summary' is cheap and "
        "should be your first call for open-ended questions like 'how is "
        "yield today'. Use 'date_range' to pull rows in a time window, "
        "'failed_only' to look at failures, and 'wafer_range' to drill "
        "into specific wafers. Timestamps must be ISO 8601 strings such "
        "as '2026-04-01 14:00:00'. The dataset covers a single 24 hour "
        "production day."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": sorted(VALID_QUERY_TYPES),
                "description": (
                    "summary returns aggregate counts, yield, and a "
                    "per-hour breakdown. date_range returns rows whose "
                    "timestamp falls in [start_time, end_time). "
                    "failed_only returns only FAIL rows, optionally "
                    "bounded by time. wafer_range returns rows for a "
                    "list of wafer_ids."
                ),
            },
            "start_time": {
                "type": "string",
                "description": "Inclusive ISO 8601 timestamp lower bound.",
            },
            "end_time": {
                "type": "string",
                "description": "Exclusive ISO 8601 timestamp upper bound.",
            },
            "wafer_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of wafer identifiers, for example ['W050', 'W051'].",
            },
        },
        "required": ["query_type"],
    },
}


CALCULATE_SPC_METRICS_SCHEMA: dict[str, Any] = {
    "name": "calculate_spc_metrics",
    "description": (
        "Compute Statistical Process Control metrics for a single chip "
        "test metric over a time window. Returns the mean, sample "
        "standard deviation, upper and lower control limits at "
        "mean +/- 3 sigma, the per-subgroup means, and the subgroups "
        "whose mean fell outside the control limits. Use this when you "
        "want to inspect the stability of one metric over time, for "
        "example to see whether NPU TOPS drifts in the afternoon. Pair "
        "with generate_chart later to render the control chart."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": sorted(VALID_METRICS),
                "description": (
                    "The chip-level metric to analyze. One of npu_tops, "
                    "npu_power_w, cpu_freq_ghz, memory_bandwidth_gbps, "
                    "die_temp_c."
                ),
            },
            "start_time": {
                "type": "string",
                "description": "Inclusive ISO 8601 timestamp lower bound.",
            },
            "end_time": {
                "type": "string",
                "description": "Exclusive ISO 8601 timestamp upper bound.",
            },
            "group_by": {
                "type": "string",
                "enum": sorted(VALID_GROUP_BY),
                "description": (
                    "How to subgroup the data: 'hour' aggregates by "
                    "calendar hour; 'wafer_id' aggregates by wafer."
                ),
            },
        },
        "required": ["metric", "start_time", "end_time", "group_by"],
    },
}


DETECT_ANOMALIES_SCHEMA: dict[str, Any] = {
    "name": "detect_anomalies",
    "description": (
        "Find time windows (hours) where the chip failure rate exceeded "
        "a threshold and rank test metrics by their Pearson correlation "
        "with the hourly failure rate. Use this for 'why did yield drop' "
        "questions: anomalous_windows tells you when, correlations tells "
        "you which subsystem moved with the failures. A strongly "
        "negative correlation on npu_tops paired with a strongly "
        "positive correlation on npu_power_w fingerprints an NPU "
        "power-domain excursion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start_time": {
                "type": "string",
                "description": "Inclusive ISO 8601 timestamp lower bound.",
            },
            "end_time": {
                "type": "string",
                "description": "Exclusive ISO 8601 timestamp upper bound.",
            },
            "failure_rate_threshold": {
                "type": "number",
                "description": (
                    "Hours whose failure rate is strictly greater than "
                    "this threshold are flagged as anomalous. Range 0 "
                    "to 1. Default 0.10 (10 percent)."
                ),
            },
        },
        "required": ["start_time", "end_time"],
    },
}


GENERATE_CHART_SCHEMA: dict[str, Any] = {
    "name": "generate_chart",
    "description": (
        "Render a matplotlib chart for the agent to attach to its answer "
        "and return the absolute path to the saved PNG. Three chart "
        "types are supported. 'spc_chart' plots one metric's hourly "
        "mean over time with the mean and +/- 3 sigma control limits, "
        "highlighting any out-of-control hours in red. "
        "'correlation_chart' plots two metrics on dual y-axes and shows "
        "the Pearson correlation between them in the title. "
        "'failure_timeline' scatters every failed chip in the window, "
        "with the y-axis labeled by failure_reason. Use a chart to give "
        "a yield engineer something concrete to look at when answering "
        "investigative questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": sorted(VALID_CHART_TYPES),
                "description": (
                    "Which chart to draw. spc_chart requires "
                    "primary_metric; correlation_chart requires both "
                    "primary_metric and secondary_metric; "
                    "failure_timeline requires neither."
                ),
            },
            "primary_metric": {
                "type": "string",
                "enum": sorted(VALID_METRICS),
                "description": (
                    "Required for spc_chart and correlation_chart. "
                    "Ignored for failure_timeline."
                ),
            },
            "secondary_metric": {
                "type": "string",
                "enum": sorted(VALID_METRICS),
                "description": (
                    "Required for correlation_chart, must differ from "
                    "primary_metric. Ignored otherwise."
                ),
            },
            "start_time": {
                "type": "string",
                "description": "Inclusive ISO 8601 timestamp lower bound.",
            },
            "end_time": {
                "type": "string",
                "description": "Exclusive ISO 8601 timestamp upper bound.",
            },
        },
        "required": ["chart_type", "start_time", "end_time"],
    },
}


WRITE_SUMMARY_REPORT_SCHEMA: dict[str, Any] = {
    "name": "write_summary_report",
    "description": (
        "Render the final structured markdown report. This report is "
        "the engineer's deliverable: render the findings as a flowing "
        "narrative that walks through what happened, then state the "
        "root cause hypothesis, then the recommendations. Use this as "
        "the final synthesis step after you have queried the data, "
        "computed metrics, and generated any charts. Embed "
        "{{chart:chart_type:primary_metric}} (or "
        "{{chart:chart_type:primary_metric:secondary_metric}}) tokens "
        "inside a finding's description on their own line when a chart "
        "you generated visually reinforces that specific finding; the "
        "UI expands each token into the rendered chart at that position. "
        "Skip the token when a chart would not add information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": (
                    "Ordered list of categorized findings with evidence. "
                    "Each finding has a short category label, a "
                    "plain-English description, and an evidence string "
                    "that cites concrete numbers from earlier tool "
                    "results. Findings should read together as a "
                    "coherent narrative, not as disconnected bullets."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": (
                                "Short label, for example 'NPU performance', "
                                "'Yield drop', 'Power domain'."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "One or two short paragraphs describing "
                                "the finding. May contain a "
                                "{{chart:...}} token on its own line "
                                "where a chart visually reinforces the "
                                "claim."
                            ),
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Concrete numeric evidence from earlier tool results.",
                        },
                    },
                    "required": ["category", "description", "evidence"],
                },
            },
            "root_cause_hypothesis": {
                "type": "string",
                "description": (
                    "One short paragraph (two to four sentences) that "
                    "states what the evidence converges on and names "
                    "the affected subsystem (NPU, CPU, memory, "
                    "thermal). Sits between Findings and Recommendations."
                ),
            },
            "recommendations": {
                "type": "array",
                "description": "Concrete next actions for the engineer, in priority order.",
                "items": {"type": "string"},
            },
        },
        "required": ["findings", "root_cause_hypothesis", "recommendations"],
    },
}


# ---------------------------------------------------------------------------
# Public registries
# ---------------------------------------------------------------------------

# What Claude sees: a list of tool schemas.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    QUERY_DATABASE_SCHEMA,
    CALCULATE_SPC_METRICS_SCHEMA,
    DETECT_ANOMALIES_SCHEMA,
    GENERATE_CHART_SCHEMA,
    WRITE_SUMMARY_REPORT_SCHEMA,
]

# What the agent loop runs locally: a name -> callable map.
TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "query_database": query_database,
    "calculate_spc_metrics": calculate_spc_metrics,
    "detect_anomalies": detect_anomalies,
    "generate_chart": generate_chart,
    "write_summary_report": write_summary_report,
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool_use block to its Python implementation.

    The agent loop calls this function with the `name` and `input` fields
    from a Claude tool_use content block. We catch any error and return
    it as a structured payload so Claude can read it and try a different
    approach.
    """
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"unknown tool: {name}"}

    try:
        return impl(**tool_input)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001  (we log everything for the agent)
        return {"error": f"unhandled error in {name}: {type(exc).__name__}: {exc}"}


def serialize_tool_result(result: dict[str, Any]) -> str:
    """Convert a tool result dict to the JSON string we send back to Claude."""
    return json.dumps(result, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
