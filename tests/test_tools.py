"""
Unit tests for the Snapdragon Yield Analytics tool catalog.

Tests run against a fresh deterministic SQLite database built from seed 42
in the `chip_db` fixture defined in conftest.py.
"""

from __future__ import annotations

import json
from pathlib import Path

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

def test_summary_drift_day(chip_db):
    """Query the drift day (last day in the dataset)."""
    result = tools.query_database(
        query_type="summary",
        start_time="2026-04-07 00:00:00",
        end_time="2026-04-08 00:00:00",
        db_path=chip_db,
    )
    assert result["query_type"] == "summary"
    assert result["row_count"] == 10_000

    s = result["summary"]
    assert s["total_chips"] == 10_000
    assert s["passed"] + s["failed"] == 10_000
    assert 0.80 <= s["yield"] <= 0.90  # drift-day yield lands in this band by design

    # Hourly slice covers all 24 hours of the drift day.
    assert len(s["hourly_yield"]) == 24
    # Drift hours must have noticeably lower yield than morning hours.
    morning = [h for h in s["hourly_yield"] if h["hour"].endswith(" 08")]
    afternoon = [h for h in s["hourly_yield"] if h["hour"].endswith(" 17")]
    assert morning and afternoon
    assert morning[0]["yield"] > afternoon[0]["yield"]


def test_summary_for_drift_window(chip_db):
    """Drift hours of the drift day should yield 60-75 percent."""
    result = tools.query_database(
        query_type="summary",
        start_time="2026-04-07 14:00:00",
        end_time="2026-04-08 00:00:00",
        db_path=chip_db,
    )
    s = result["summary"]
    assert 0.60 <= s["yield"] <= 0.75


def test_summary_no_window_covers_all_days(chip_db):
    """A summary call with no window returns the full multi-day dataset.

    For multi-day windows the summary keeps tool responses compact:
    daily_yield carries the per-day rollup, hourly_yield is capped to
    the most recent day's 24 entries so the agent's context does not
    bloat.
    """
    result = tools.query_database(query_type="summary", db_path=chip_db)
    s = result["summary"]
    assert s["total_chips"] == 70_000
    # Six clean days at ~97-98 percent and one drift day at ~85 percent
    # average to roughly 96 percent overall.
    assert 0.94 <= s["yield"] <= 0.97
    # daily_yield has one entry per day in the window.
    assert len(s["daily_yield"]) == 7
    # hourly_yield is now capped to the most recent day's 24 hours.
    assert len(s["hourly_yield"]) == 24
    # The last day in daily_yield is the drift day (~85 percent yield).
    last_day = s["daily_yield"][-1]
    assert 0.80 <= last_day["yield"] <= 0.90


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
    # 7 days * 10,000 chips per day.
    assert decoded["summary"]["total_chips"] == 70_000


# ---------------------------------------------------------------------------
# calculate_spc_metrics: validation
# ---------------------------------------------------------------------------

def test_spc_rejects_unknown_metric(chip_db):
    with pytest.raises(ValueError):
        tools.calculate_spc_metrics(
            metric="bogus",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            group_by="hour",
            db_path=chip_db,
        )


def test_spc_rejects_unknown_group_by(chip_db):
    with pytest.raises(ValueError):
        tools.calculate_spc_metrics(
            metric="npu_tops",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            group_by="day",
            db_path=chip_db,
        )


def test_spc_requires_time_window(chip_db):
    with pytest.raises(ValueError):
        tools.calculate_spc_metrics(
            metric="npu_tops",
            start_time=None,
            end_time=None,
            group_by="hour",
            db_path=chip_db,
        )


# ---------------------------------------------------------------------------
# calculate_spc_metrics: numeric correctness
# ---------------------------------------------------------------------------

def test_spc_limits_match_mean_plus_minus_three_sigma(chip_db):
    result = tools.calculate_spc_metrics(
        metric="npu_tops",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        group_by="hour",
        db_path=chip_db,
    )
    # Brief calls for limits at mean +/- 3 sigma exactly. Floats are
    # rounded to 4 decimals, so allow a tiny rounding tolerance.
    assert abs((result["ucl"] - result["mean"]) - 3 * result["std"]) < 1e-3
    assert abs((result["mean"] - result["lcl"]) - 3 * result["std"]) < 1e-3


def test_spc_hourly_returns_24_groups(chip_db):
    result = tools.calculate_spc_metrics(
        metric="npu_tops",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        group_by="hour",
        db_path=chip_db,
    )
    assert len(result["groups"]) == 24
    for g in result["groups"]:
        assert g["n"] > 0
        assert isinstance(g["mean"], float)


def test_spc_wafer_drift_lots_have_lower_npu_tops_mean(chip_db):
    """On the drift day (2026-04-07) wafers are W600..W699. Late wafers
    fall in drift hours and should average noticeably lower NPU TOPS
    than early wafers."""
    result = tools.calculate_spc_metrics(
        metric="npu_tops",
        start_time="2026-04-07 00:00:00",
        end_time="2026-04-08 00:00:00",
        group_by="wafer_id",
        db_path=chip_db,
    )
    assert len(result["groups"]) == 100
    by_wafer = {g["group"]: g["mean"] for g in result["groups"]}
    assert by_wafer["W605"] > by_wafer["W695"]


def test_spc_schema_in_list():
    names = [s["name"] for s in tools.TOOL_SCHEMAS]
    assert "calculate_spc_metrics" in names


# ---------------------------------------------------------------------------
# detect_anomalies: validation
# ---------------------------------------------------------------------------

def test_anomalies_rejects_threshold_above_one(chip_db):
    with pytest.raises(ValueError):
        tools.detect_anomalies(
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            failure_rate_threshold=1.5,
            db_path=chip_db,
        )


def test_anomalies_requires_time_window(chip_db):
    with pytest.raises(ValueError):
        tools.detect_anomalies(
            start_time=None,
            end_time=None,
            db_path=chip_db,
        )


# ---------------------------------------------------------------------------
# detect_anomalies: behavior
# ---------------------------------------------------------------------------

def test_anomalies_default_threshold_flags_drift_hours(chip_db):
    """Drift is on day 7 (2026-04-07). The default threshold should
    flag every hour of that day's afternoon."""
    result = tools.detect_anomalies(
        start_time="2026-04-07 00:00:00",
        end_time="2026-04-08 00:00:00",
        db_path=chip_db,
    )
    flagged = {row["hour"] for row in result["anomalous_windows"]}
    drift_hours = {f"2026-04-07 {h:02d}" for h in range(14, 24)}
    assert drift_hours.issubset(flagged)
    # Mid-morning hour 03 has yield > 95 percent and should not be flagged.
    assert "2026-04-07 03" not in flagged


def test_anomalies_high_threshold_flags_nothing(chip_db):
    result = tools.detect_anomalies(
        start_time="2026-04-07 00:00:00",
        end_time="2026-04-08 00:00:00",
        failure_rate_threshold=0.99,
        db_path=chip_db,
    )
    assert result["anomalous_windows"] == []


def test_anomalies_correlations_match_storyline(chip_db):
    """Correlations on the drift day should fingerprint the NPU power-domain
    excursion."""
    result = tools.detect_anomalies(
        start_time="2026-04-07 00:00:00",
        end_time="2026-04-08 00:00:00",
        db_path=chip_db,
    )
    corrs = result["correlations"]
    # NPU TOPS hourly mean falls during drift while failure rate rises:
    # strongly negative.
    assert corrs["npu_tops"]["r"] < -0.7
    # NPU power hourly mean rises while failure rate rises: strongly positive.
    assert corrs["npu_power_w"]["r"] > 0.7
    # CPU, memory bandwidth, die temp stay normal across drift; correlation
    # with failure rate should be weak.
    assert abs(corrs["cpu_freq_ghz"]["r"]) < 0.6
    assert abs(corrs["memory_bandwidth_gbps"]["r"]) < 0.6
    assert abs(corrs["die_temp_c"]["r"]) < 0.6


def test_anomalies_overall_failure_rate_matches_summary(chip_db):
    """Cross-tool consistency check on the same day window."""
    window = {
        "start_time": "2026-04-07 00:00:00",
        "end_time": "2026-04-08 00:00:00",
    }
    a = tools.detect_anomalies(db_path=chip_db, **window)
    s = tools.query_database(query_type="summary", db_path=chip_db, **window)["summary"]
    assert a["n_total"] == s["total_chips"]
    assert a["n_failed"] == s["failed"]


def test_anomalies_schema_in_list():
    names = [s["name"] for s in tools.TOOL_SCHEMAS]
    assert "detect_anomalies" in names


# ---------------------------------------------------------------------------
# generate_chart: validation
# ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path):
    with open(path, "rb") as f:
        return f.read(8) == PNG_MAGIC


def test_chart_rejects_unknown_type(chip_db, tmp_path):
    with pytest.raises(ValueError):
        tools.generate_chart(
            chart_type="bar_chart",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            primary_metric="npu_tops",
            output_dir=tmp_path,
            db_path=chip_db,
        )


def test_chart_requires_time_window(chip_db, tmp_path):
    with pytest.raises(ValueError):
        tools.generate_chart(
            chart_type="spc_chart",
            start_time=None,
            end_time=None,
            primary_metric="npu_tops",
            output_dir=tmp_path,
            db_path=chip_db,
        )


def test_chart_spc_requires_primary_metric(chip_db, tmp_path):
    with pytest.raises(ValueError):
        tools.generate_chart(
            chart_type="spc_chart",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            primary_metric=None,
            output_dir=tmp_path,
            db_path=chip_db,
        )


def test_chart_spc_rejects_unknown_metric(chip_db, tmp_path):
    with pytest.raises(ValueError):
        tools.generate_chart(
            chart_type="spc_chart",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            primary_metric="bogus_metric",
            output_dir=tmp_path,
            db_path=chip_db,
        )


def test_chart_correlation_requires_both_metrics(chip_db, tmp_path):
    with pytest.raises(ValueError):
        tools.generate_chart(
            chart_type="correlation_chart",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            primary_metric="npu_tops",
            secondary_metric=None,
            output_dir=tmp_path,
            db_path=chip_db,
        )


def test_chart_correlation_rejects_same_metric_twice(chip_db, tmp_path):
    with pytest.raises(ValueError):
        tools.generate_chart(
            chart_type="correlation_chart",
            start_time="2026-04-01 00:00:00",
            end_time="2026-04-02 00:00:00",
            primary_metric="npu_tops",
            secondary_metric="npu_tops",
            output_dir=tmp_path,
            db_path=chip_db,
        )


# ---------------------------------------------------------------------------
# generate_chart: each chart type produces a valid PNG
# ---------------------------------------------------------------------------

def test_chart_spc_produces_png(chip_db, tmp_path):
    result = tools.generate_chart(
        chart_type="spc_chart",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        primary_metric="npu_tops",
        output_dir=tmp_path,
        db_path=chip_db,
    )
    path = Path(result["path"])
    assert path.exists()
    assert path.is_absolute()
    assert path.stat().st_size > 0
    assert _is_png(path)
    assert result["chart_type"] == "spc_chart"
    assert result["primary_metric"] == "npu_tops"
    assert result["filename"].startswith("spc_npu_tops_")


def test_chart_correlation_produces_png(chip_db, tmp_path):
    result = tools.generate_chart(
        chart_type="correlation_chart",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        primary_metric="npu_tops",
        secondary_metric="npu_power_w",
        output_dir=tmp_path,
        db_path=chip_db,
    )
    path = Path(result["path"])
    assert path.exists()
    assert _is_png(path)
    assert result["secondary_metric"] == "npu_power_w"
    assert "npu_tops_vs_npu_power_w" in result["filename"]


def test_chart_failure_timeline_produces_png(chip_db, tmp_path):
    result = tools.generate_chart(
        chart_type="failure_timeline",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        output_dir=tmp_path,
        db_path=chip_db,
    )
    path = Path(result["path"])
    assert path.exists()
    assert _is_png(path)
    assert result["filename"].startswith("failure_timeline_")


def test_chart_failure_timeline_handles_window_with_no_failures(chip_db, tmp_path):
    # Tiny early-morning window where the failure rate is very low; even if
    # this window has zero failures the chart should still render with a
    # placeholder rather than crashing.
    result = tools.generate_chart(
        chart_type="failure_timeline",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-01 00:01:00",
        output_dir=tmp_path,
        db_path=chip_db,
    )
    assert _is_png(Path(result["path"]))


def test_chart_uses_output_dir_override(chip_db, tmp_path):
    result = tools.generate_chart(
        chart_type="spc_chart",
        start_time="2026-04-01 00:00:00",
        end_time="2026-04-02 00:00:00",
        primary_metric="npu_tops",
        output_dir=tmp_path,
        db_path=chip_db,
    )
    assert Path(result["path"]).parent == tmp_path.resolve()


def test_chart_schema_in_list():
    names = [s["name"] for s in tools.TOOL_SCHEMAS]
    assert "generate_chart" in names


# ---------------------------------------------------------------------------
# write_summary_report: validation
# ---------------------------------------------------------------------------

_VALID_FINDING = {
    "category": "NPU performance",
    "description": (
        "NPU TOPS dropped from 50.5 to 47.8 starting at 14:00; the "
        "hourly mean fell to 47.6 by 17:00 across 416 chips."
    ),
}

_VALID_ROOT_CAUSE = (
    "Hexagon NPU subsystem excursion starting at 14:00 dragged TOPS "
    "below spec while NPU power rose; CPU, memory, and thermal stayed "
    "within control limits."
)

_VALID_RECOMMENDATIONS = [
    "Quarantine afternoon-shift wafers pending secondary validation.",
    "Re-run SPC grouped by wafer_id for the drift window.",
]


def test_report_rejects_non_list_findings():
    with pytest.raises(ValueError):
        tools.write_summary_report(
            findings="oops",  # type: ignore[arg-type]
            root_cause_hypothesis=_VALID_ROOT_CAUSE,
            recommendations=_VALID_RECOMMENDATIONS,
        )


def test_report_rejects_finding_missing_field():
    bad = [{"category": "x"}]  # no description
    with pytest.raises(ValueError):
        tools.write_summary_report(
            findings=bad,
            root_cause_hypothesis=_VALID_ROOT_CAUSE,
            recommendations=_VALID_RECOMMENDATIONS,
        )


def test_report_rejects_finding_field_wrong_type():
    bad = [{"category": "x", "description": 123}]
    with pytest.raises(ValueError):
        tools.write_summary_report(
            findings=bad,
            root_cause_hypothesis=_VALID_ROOT_CAUSE,
            recommendations=_VALID_RECOMMENDATIONS,
        )


def test_report_rejects_empty_root_cause_hypothesis():
    with pytest.raises(ValueError):
        tools.write_summary_report(
            findings=[_VALID_FINDING],
            root_cause_hypothesis="   ",
            recommendations=_VALID_RECOMMENDATIONS,
        )


def test_report_rejects_non_list_recommendations():
    with pytest.raises(ValueError):
        tools.write_summary_report(
            findings=[_VALID_FINDING],
            root_cause_hypothesis=_VALID_ROOT_CAUSE,
            recommendations="not a list",  # type: ignore[arg-type]
        )


def test_report_rejects_non_string_recommendation():
    with pytest.raises(ValueError):
        tools.write_summary_report(
            findings=[_VALID_FINDING],
            root_cause_hypothesis=_VALID_ROOT_CAUSE,
            recommendations=["fine", 42],  # type: ignore[list-item]
        )


# ---------------------------------------------------------------------------
# write_summary_report: rendered output
# ---------------------------------------------------------------------------

def test_report_contains_all_sections():
    result = tools.write_summary_report(
        findings=[_VALID_FINDING],
        root_cause_hypothesis=_VALID_ROOT_CAUSE,
        recommendations=_VALID_RECOMMENDATIONS,
    )
    md = result["report"]
    # The report no longer carries an H1 title; the italic generated-on
    # line is the soft intro and the H2 sections carry the structure.
    assert "# Yield Analysis Report" not in md
    assert "## Findings" in md
    assert "## Root cause hypothesis" in md
    assert "## Bottom line" not in md
    assert "## Recommendations" in md
    assert "Snapdragon production data" in md


def test_report_renders_finding_fields():
    result = tools.write_summary_report(
        findings=[_VALID_FINDING],
        root_cause_hypothesis=_VALID_ROOT_CAUSE,
        recommendations=_VALID_RECOMMENDATIONS,
    )
    md = result["report"]
    assert _VALID_FINDING["category"] in md
    assert _VALID_FINDING["description"] in md
    assert "_Evidence:" not in md


def test_report_numbers_recommendations():
    result = tools.write_summary_report(
        findings=[_VALID_FINDING],
        root_cause_hypothesis=_VALID_ROOT_CAUSE,
        recommendations=_VALID_RECOMMENDATIONS,
    )
    md = result["report"]
    assert "1. " + _VALID_RECOMMENDATIONS[0] in md
    assert "2. " + _VALID_RECOMMENDATIONS[1] in md


def test_report_counts_match_inputs():
    findings = [_VALID_FINDING, _VALID_FINDING, _VALID_FINDING]
    recs = ["a", "b"]
    result = tools.write_summary_report(
        findings=findings,
        root_cause_hypothesis=_VALID_ROOT_CAUSE,
        recommendations=recs,
    )
    assert result["n_findings"] == 3
    assert result["n_recommendations"] == 2
    assert result["char_count"] == len(result["report"])


def test_report_handles_empty_findings_and_recommendations():
    result = tools.write_summary_report(
        findings=[],
        root_cause_hypothesis=_VALID_ROOT_CAUSE,
        recommendations=[],
    )
    md = result["report"]
    assert "## Findings" not in md
    assert "## Recommendations" not in md
    assert "## Root cause hypothesis" in md


def test_report_schema_in_list():
    names = [s["name"] for s in tools.TOOL_SCHEMAS]
    assert "write_summary_report" in names


def test_report_serialize_round_trips():
    result = tools.write_summary_report(
        findings=[_VALID_FINDING],
        root_cause_hypothesis=_VALID_ROOT_CAUSE,
        recommendations=_VALID_RECOMMENDATIONS,
    )
    encoded = tools.serialize_tool_result(result)
    decoded = json.loads(encoded)
    assert decoded["report"] == result["report"]
