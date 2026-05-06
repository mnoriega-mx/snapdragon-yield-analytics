"""
Tests for the synthetic Snapdragon dataset generator.

These tests cover the contract from section 10 of the project brief:
    * The dataset has exactly 10,000 rows
    * All required columns are present
    * Normal hours (00:00 to 13:59) yield above 90 percent
    * Drift hours  (14:00 to 23:59) yield between 60 and 75 percent
    * The fixed random seed produces identical output across runs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable so `import data.generate_data` works when
# pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_data import (  # noqa: E402  (import after sys.path tweak)
    DRIFT_START_HOUR,
    TOTAL_CHIPS,
    generate_dataset,
)

REQUIRED_COLUMNS = {
    "timestamp",
    "wafer_id",
    "chip_id",
    "soc_model",
    "process_node",
    "npu_tops",
    "npu_power_w",
    "cpu_freq_ghz",
    "memory_bandwidth_gbps",
    "die_temp_c",
    "test_result",
    "failure_reason",
}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return generate_dataset(seed=42)


def test_row_count(df: pd.DataFrame) -> None:
    assert len(df) == TOTAL_CHIPS == 10_000


def test_columns_present(df: pd.DataFrame) -> None:
    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_no_unexpected_columns(df: pd.DataFrame) -> None:
    assert set(df.columns) == REQUIRED_COLUMNS


def test_pass_fail_only(df: pd.DataFrame) -> None:
    assert set(df["test_result"].unique()).issubset({"PASS", "FAIL"})


def test_failure_reason_only_when_failed(df: pd.DataFrame) -> None:
    passed_with_reason = df[
        (df["test_result"] == "PASS") & df["failure_reason"].notna()
    ]
    assert len(passed_with_reason) == 0, "PASS rows must not have a failure_reason"

    failed_without_reason = df[
        (df["test_result"] == "FAIL") & df["failure_reason"].isna()
    ]
    assert len(failed_without_reason) == 0, "FAIL rows must have a failure_reason"


def test_normal_hour_yield_above_90_percent(df: pd.DataFrame) -> None:
    hours = df["timestamp"].dt.hour
    normal = df[hours < DRIFT_START_HOUR]
    yield_pct = (normal["test_result"] == "PASS").mean()
    assert yield_pct > 0.90, f"normal-hour yield was {yield_pct:.1%}"


def test_drift_hour_yield_in_band(df: pd.DataFrame) -> None:
    hours = df["timestamp"].dt.hour
    drift = df[hours >= DRIFT_START_HOUR]
    yield_pct = (drift["test_result"] == "PASS").mean()
    assert 0.60 <= yield_pct <= 0.75, f"drift-hour yield was {yield_pct:.1%}, expected 60 to 75 percent"


def test_seed_is_reproducible() -> None:
    a = generate_dataset(seed=42)
    b = generate_dataset(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_output() -> None:
    a = generate_dataset(seed=42)
    b = generate_dataset(seed=1234)
    # The numeric metric columns should differ when the seed changes.
    assert not a["npu_tops"].equals(b["npu_tops"])


def test_all_chips_are_sd8gen5_3nm(df: pd.DataFrame) -> None:
    assert (df["soc_model"] == "SD8Gen5").all()
    assert (df["process_node"] == "3nm").all()


def test_drift_hour_failed_chips_match_narrative(df: pd.DataFrame) -> None:
    """The agent's storyline says failed chips average ~42 TOPS and ~4.5W.

    This guards against future generator tweaks accidentally undoing the
    NPU-domain drift signal.
    """
    hours = df["timestamp"].dt.hour
    failed_drift = df[(hours >= DRIFT_START_HOUR) & (df["test_result"] == "FAIL")]
    assert len(failed_drift) > 0
    assert failed_drift["npu_tops"].mean() < 45, (
        f"failed drift-hour chips averaged {failed_drift['npu_tops'].mean():.2f} TOPS"
    )
    assert failed_drift["npu_power_w"].mean() > 4.0, (
        f"failed drift-hour chips averaged {failed_drift['npu_power_w'].mean():.2f} W"
    )
