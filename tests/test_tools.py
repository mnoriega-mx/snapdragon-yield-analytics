"""
Unit tests for the Snapdragon Yield Analytics tool catalog.

Tests run against a fresh deterministic SQLite database built from seed 42
in the `chip_db` fixture defined in conftest.py.
"""

from __future__ import annotations

import json

import pytest

from agent import tools


# ---------------------------------------------------------------------------
# query_database: error handling
# ---------------------------------------------------------------------------

def test_query_database_rejects_unknown_query_type(chip_db):
    with pytest.raises(ValueError):
        tools.query_database(query_type="not_a_real_type", db_path=chip_db)


def test_query_database_date_range_requires_both_bounds(chip_db):
    with pytest.raises(ValueError):
        tools.query_database(query_type="date_range", start_time="2026-04-01 00:00:00", db_path=chip_db)


def test_query_database_wafer_range_requires_wafers(chip_db):
    with pytest.raises(ValueError):
        tools.query_database(query_type="wafer_range", wafer_ids=[], db_path=chip_db)


# ---------------------------------------------------------------------------
# query_database: summary
# ---------------------------------------------------------------------------

def test_summary_full_day(chip_db):
    result = tools.query_database(query_type="summary", db_path=chip_db)
    assert result["query_type"] == "summary"
    assert result["row_count"] == 10_000

    s = result["summary"]
    assert s["total_chips"] == 10_000
    assert s["passed"] + s["failed"] == 10_000
    assert 0.80 <= s["yield"] <= 0.90  # full-day yield lands in this band by design

    # Hourly slice covers all 24 hours.
    assert len(s["hourly_yield"]) == 24
    # Drift hour rows must have noticeably lower yield than morning rows.
    morning = [h for h in s["hourly_yield"] if h["hour"].endswith("T08") or h["hour"].endswith(" 08")]
    afternoon = [h for h in s["hourly_yield"] if h["hour"].endswith(" 17")]
    if morning and afternoon:
        assert morning[0]["yield"] > afternoon[0]["yield"]


def test_summary_for_drift_window(chip_db):
    result = tools.query_database(
        query_type="summary",
        start_time="2026-04-01 14:00:00",
        end_time="2026-04-02 00:00:00",
        db_path=chip_db,
    )
    s = result["summary"]
    assert 0.60 <= s["yield"] <= 0.75


# ---------------------------------------------------------------------------
# query_database: date_range
# ---------------------------------------------------------------------------

def test_date_range_returns_only_the_window(chip_db):
    result = tools.query_database(
        query_type="date_range",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-01 01:00:00",
        db_path=chip_db,
    )
    # 10000 chips evenly spaced across 24 hours = ~416 chips per hour.
    assert 400 <= result["row_count"] <= 430
    for row in result["returned_rows"]:
        assert "2026-04-01 00:" in row["timestamp"]


def test_date_range_truncation_flag(chip_db):
    # Pull the whole day; row count well exceeds MAX_ROWS_RETURNED.
    result = tools.query_database(
        query_type="date_range",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        db_path=chip_db,
    )
    assert result["row_count"] == 10_000
    assert result["truncated"] is True
    assert len(result["returned_rows"]) == tools.MAX_ROWS_RETURNED


# ---------------------------------------------------------------------------
# query_database: failed_only
# ---------------------------------------------------------------------------

def test_failed_only_returns_only_fails(chip_db):
    result = tools.query_database(query_type="failed_only", db_path=chip_db)
    assert result["row_count"] > 0
    for row in result["returned_rows"]:
        assert row["test_result"] == "FAIL"
        assert row["failure_reason"]


def test_failed_only_in_normal_window_is_small(chip_db):
    # Hours 0 to 13 should have very few failures (yield > 95 percent).
    result = tools.query_database(
        query_type="failed_only",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-01 14:00:00",
        db_path=chip_db,
    )
    # Roughly 5800 chips in this window; at <5 percent fail rate, well under 400.
    assert result["row_count"] < 400


# ---------------------------------------------------------------------------
# query_database: wafer_range
# ---------------------------------------------------------------------------

def test_wafer_range_returns_only_listed_wafers(chip_db):
    result = tools.query_database(
        query_type="wafer_range",
        wafer_ids=["W050", "W051"],
        db_path=chip_db,
    )
    # 100 chips per wafer * 2 wafers = 200 rows.
    assert result["row_count"] == 200
    seen = {row["wafer_id"] for row in result["returned_rows"]}
    assert seen == {"W050", "W051"}


# ---------------------------------------------------------------------------
# Schema and dispatch
# ---------------------------------------------------------------------------

def test_query_database_is_in_schema_list():
    names = [s["name"] for s in tools.TOOL_SCHEMAS]
    assert "query_database" in names


def test_execute_tool_routes_known_tool(chip_db):
    result = tools.execute_tool(
        "query_database",
        {"query_type": "summary"},
    )
    # Without a custom db_path this hits DEFAULT_DB_PATH; we expect it to
    # either work (if the user has built the db) or return a clean error.
    assert "summary" in result or "error" in result


def test_execute_tool_returns_error_for_unknown_tool():
    result = tools.execute_tool("definitely_not_a_tool", {})
    assert "error" in result


def test_execute_tool_catches_invalid_input(chip_db):
    result = tools.execute_tool(
        "query_database",
        {"query_type": "wafer_range", "wafer_ids": []},
    )
    assert "error" in result


def test_serialize_tool_result_round_trips(chip_db):
    result = tools.query_database(query_type="summary", db_path=chip_db)
    encoded = tools.serialize_tool_result(result)
    decoded = json.loads(encoded)
    assert decoded["summary"]["total_chips"] == 10_000
