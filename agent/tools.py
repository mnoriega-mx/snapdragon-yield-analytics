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
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

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
                for r in hourly
            ],
        },
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


# ---------------------------------------------------------------------------
# Public registries
# ---------------------------------------------------------------------------

# What Claude sees: a list of tool schemas.
TOOL_SCHEMAS: list[dict[str, Any]] = [QUERY_DATABASE_SCHEMA]

# What the agent loop runs locally: a name -> callable map.
TOOL_IMPLEMENTATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "query_database": query_database,
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
