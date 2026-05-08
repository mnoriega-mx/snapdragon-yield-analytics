"""
Synthetic data generator for the Snapdragon Yield Analytics demo.

Produces a CSV with N production days of fictional Snapdragon 8 Gen 5
(3nm) test results. Each day has 10,000 chips across 100 wafers, evenly
spaced across 24 hours. Default is 7 days, with the drift excursion
injected only on the LAST day so the prior six serve as a clean
baseline for the dashboard and week-over-week comparisons.

Per-day distribution parameters (every chip on every non-drift day):
    NPU TOPS                mean 50.5,  std 1.2
    NPU Power (W)           mean 3.20,  std 0.10
    CPU frequency (GHz)     mean 3.45,  std 0.05
    Memory bandwidth (GB/s) mean 205,   std 5
    Die temperature (C)     mean 78,    std 3
    Expected yield: ~95 percent

Drift, applied only to the LAST day's hours 14:00 to 23:59:
    A fraction (DRIFT_AFFECTED_FRACTION, default 0.32) of late-day
    chips exhibit an NPU-domain process excursion:
        NPU TOPS            mean 42.0,  std 2.5  (drops, more variance)
        NPU Power (W)       mean 4.5,   std 0.30 (rises sharply)
        Other metrics: identical to normal operation
    The remaining late-day chips sample from the normal distribution.
    Expected last-day drift-hour yield: ~68 percent.

Wafer ids and chip ids are unique across days. With WAFERS=100 per day,
day 1 has W000-W099, day 2 has W100-W199, etc. A chip is PASS only if
every metric meets spec. failure_reason is set to the first failed
metric (in a documented priority order) so the agent can group failures
by root cause.

Usage:
    python data/generate_data.py                      # default 7 days
    python data/generate_data.py --days 1             # single-day mode
    python data/generate_data.py --output path/to/file.csv

The script is fully deterministic for a given seed (default: 42).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Spec thresholds (used to decide PASS vs FAIL on each chip)
# ---------------------------------------------------------------------------

NPU_TOPS_MIN = 48.0           # target 50, pass at >= 48
NPU_POWER_MAX_W = 3.5         # target 3.2W, pass at <= 3.5W
CPU_FREQ_MIN_GHZ = 3.3        # target 3.4 GHz, pass at >= 3.3
MEMORY_BW_MIN_GBPS = 190.0    # target 200 GB/s, pass at >= 190
DIE_TEMP_MAX_C = 95.0         # pass at < 95C

# Order matters: when a chip fails multiple metrics we report the first one
# in this list as the failure_reason. NPU first because that is the demo
# storyline.
FAILURE_PRIORITY = [
    ("npu_tops_below_spec",       lambda r: r.npu_tops < NPU_TOPS_MIN),
    ("npu_power_above_spec",      lambda r: r.npu_power_w > NPU_POWER_MAX_W),
    ("cpu_freq_below_spec",       lambda r: r.cpu_freq_ghz < CPU_FREQ_MIN_GHZ),
    ("memory_bandwidth_low",      lambda r: r.memory_bandwidth_gbps < MEMORY_BW_MIN_GBPS),
    ("die_temp_over_threshold",   lambda r: r.die_temp_c >= DIE_TEMP_MAX_C),
]


# ---------------------------------------------------------------------------
# Distribution config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricDist:
    mean: float
    std: float


@dataclass(frozen=True)
class HourProfile:
    """Distribution parameters for a single hour of production."""
    npu_tops: MetricDist
    npu_power_w: MetricDist
    cpu_freq_ghz: MetricDist
    memory_bandwidth_gbps: MetricDist
    die_temp_c: MetricDist


NORMAL = HourProfile(
    npu_tops=MetricDist(50.5, 1.2),
    npu_power_w=MetricDist(3.20, 0.10),
    cpu_freq_ghz=MetricDist(3.45, 0.05),
    memory_bandwidth_gbps=MetricDist(205.0, 5.0),
    die_temp_c=MetricDist(78.0, 3.0),
)

DRIFT = HourProfile(
    # NPU performance degrades and power climbs. CPU, memory, thermal stay
    # in spec. This is the signal the agent should isolate.
    npu_tops=MetricDist(42.0, 2.5),
    npu_power_w=MetricDist(4.5, 0.30),
    cpu_freq_ghz=NORMAL.cpu_freq_ghz,
    memory_bandwidth_gbps=NORMAL.memory_bandwidth_gbps,
    die_temp_c=NORMAL.die_temp_c,
)


# Production timeline configuration
START_TIME = datetime(2026, 4, 1, 0, 0, 0)
HOURS_TOTAL = 24
DRIFT_START_HOUR = 14
DAYS_DEFAULT = 7
TOTAL_CHIPS = 10_000  # chips PER DAY
WAFERS = 100  # wafers PER DAY
CHIPS_PER_WAFER = TOTAL_CHIPS // WAFERS

# Fraction of drift-hour chips on the LAST day that actually exhibit
# the excursion. Tuned so the drift-hour yield lands around 68 percent
# on the drift day.
DRIFT_AFFECTED_FRACTION = 0.32


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _sample(rng: np.random.Generator, dist: MetricDist, n: int) -> np.ndarray:
    return rng.normal(loc=dist.mean, scale=dist.std, size=n)


def _classify_failure(row) -> str | None:
    for reason, predicate in FAILURE_PRIORITY:
        if predicate(row):
            return reason
    return None


def _sample_profile(rng: np.random.Generator, profile: HourProfile, n: int) -> dict[str, np.ndarray]:
    return {
        "npu_tops": _sample(rng, profile.npu_tops, n),
        "npu_power_w": _sample(rng, profile.npu_power_w, n),
        "cpu_freq_ghz": _sample(rng, profile.cpu_freq_ghz, n),
        "memory_bandwidth_gbps": _sample(rng, profile.memory_bandwidth_gbps, n),
        "die_temp_c": _sample(rng, profile.die_temp_c, n),
    }


def _generate_one_day(
    rng: np.random.Generator,
    day_offset: int,
    inject_drift: bool,
    drift_affected_fraction: float,
) -> pd.DataFrame:
    """Generate one production day's chip records.

    Wafer ids and chip ids are offset by `day_offset` so they stay
    unique across the multi-day dataset (W000-W099 on day 0, W100-W199
    on day 1, etc.). Drift is injected only when `inject_drift=True`.
    """
    seconds_total = HOURS_TOTAL * 3600
    seconds_per_chip = seconds_total / TOTAL_CHIPS
    offsets = np.arange(TOTAL_CHIPS) * seconds_per_chip
    day_start = START_TIME + timedelta(days=day_offset)
    timestamps = [day_start + timedelta(seconds=float(s)) for s in offsets]
    hours = np.array([t.hour for t in timestamps])

    # Wafer / chip identifiers. Within a day, wafers tile across the 24
    # hours so each wafer contains chips from a contiguous time window.
    # Across days, ids are globally offset by day_offset * (WAFERS or
    # TOTAL_CHIPS) so they are unique.
    wafer_idx = np.repeat(np.arange(WAFERS), CHIPS_PER_WAFER)
    chip_idx = np.arange(TOTAL_CHIPS)
    wafer_ids = [f"W{w + day_offset * WAFERS:03d}" for w in wafer_idx]
    chip_ids = [f"C{day_offset * TOTAL_CHIPS + c:05d}" for c in chip_idx]

    # Start by sampling every chip from the normal distribution.
    samples = _sample_profile(rng, NORMAL, TOTAL_CHIPS)

    if inject_drift:
        # Replace a fraction of drift-hour chips with samples from the
        # drift distribution. The mask spreads affected chips across
        # late-day wafers, mirroring an excursion that hits the NPU
        # power domain intermittently.
        drift_mask_hours = hours >= DRIFT_START_HOUR
        drift_eligible_idx = np.where(drift_mask_hours)[0]
        affected_count = int(round(len(drift_eligible_idx) * drift_affected_fraction))
        affected_idx = rng.choice(drift_eligible_idx, size=affected_count, replace=False)
        affected_idx.sort()

        drift_samples = _sample_profile(rng, DRIFT, affected_count)
        for key, arr in drift_samples.items():
            samples[key][affected_idx] = arr

    df = pd.DataFrame({
        "timestamp": timestamps,
        "wafer_id": wafer_ids,
        "chip_id": chip_ids,
        "soc_model": "SD8Gen5",
        "process_node": "3nm",
        "npu_tops": np.round(samples["npu_tops"], 2),
        "npu_power_w": np.round(samples["npu_power_w"], 3),
        "cpu_freq_ghz": np.round(samples["cpu_freq_ghz"], 3),
        "memory_bandwidth_gbps": np.round(samples["memory_bandwidth_gbps"], 1),
        "die_temp_c": np.round(samples["die_temp_c"], 1),
    })

    # Decide pass/fail using the rounded values so any downstream
    # consumer of the CSV would compute the same result we did.
    failure_reasons = df.apply(_classify_failure, axis=1)
    df["test_result"] = np.where(failure_reasons.isna(), "PASS", "FAIL")
    df["failure_reason"] = failure_reasons

    return df


def generate_dataset(
    seed: int = 42,
    days: int = DAYS_DEFAULT,
    drift_affected_fraction: float = DRIFT_AFFECTED_FRACTION,
) -> pd.DataFrame:
    """Return a deterministic synthetic Snapdragon test dataset across N days.

    Drift is injected only on the LAST day so the prior days serve as a
    clean baseline that the dashboard and the agent can compare against.

    Args:
        seed: Random seed for reproducibility.
        days: Number of production days to generate. Default 7. Pass 1
            to get the original single-day behavior.
        drift_affected_fraction: Fraction of drift-hour chips on the
            last day that exhibit the NPU excursion. Default tuned to
            produce ~68 percent drift-hour yield on that day.
    """
    if days < 1:
        raise ValueError("days must be at least 1")

    rng = np.random.default_rng(seed)
    daily_dfs: list[pd.DataFrame] = []
    for day_offset in range(days):
        is_last_day = (day_offset == days - 1)
        daily_dfs.append(
            _generate_one_day(
                rng=rng,
                day_offset=day_offset,
                inject_drift=is_last_day,
                drift_affected_fraction=drift_affected_fraction,
            )
        )

    return pd.concat(daily_dfs, ignore_index=True)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Format timestamps as ISO 8601 for SQLite compatibility.
    df_out = df.copy()
    df_out["timestamp"] = df_out["timestamp"].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
    df_out.to_csv(output_path, index=False)


def _print_summary(df: pd.DataFrame) -> None:
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts = df["timestamp"]
    else:
        ts = pd.to_datetime(df["timestamp"])

    days_in_data = sorted(ts.dt.date.unique())
    pass_rate_total = (df["test_result"] == "PASS").mean()
    print(f"Generated {len(df):,} chip records across {len(days_in_data)} day(s).")
    print(f"  Overall yield: {pass_rate_total:.1%}")
    print()
    print("Yield by day:")
    df_with_day = df.assign(_day=ts.dt.date)
    for day in days_in_data:
        day_df = df_with_day[df_with_day["_day"] == day]
        yield_pct = (day_df["test_result"] == "PASS").mean()
        print(f"  {day}  yield={yield_pct:.1%}  n={len(day_df):,}")
    print()

    # Last-day breakdown so the drift signal is visible at a glance.
    last_day = days_in_data[-1]
    last = df_with_day[df_with_day["_day"] == last_day]
    last_hours = ts.loc[last.index].dt.hour
    last_normal = last.loc[last_hours < DRIFT_START_HOUR, "test_result"]
    last_drift = last.loc[last_hours >= DRIFT_START_HOUR, "test_result"]
    print(f"Last day ({last_day}) breakdown:")
    print(
        f"  Hours 00 to 13 (normal): yield="
        f"{(last_normal == 'PASS').mean():.1%}  n={len(last_normal):,}"
    )
    print(
        f"  Hours 14 to 23 (drift) : yield="
        f"{(last_drift == 'PASS').mean():.1%}  n={len(last_drift):,}"
    )
    print()
    print("Top failure reasons (all days):")
    fr_counts = df.loc[df["test_result"] == "FAIL", "failure_reason"].value_counts()
    for reason, count in fr_counts.items():
        print(f"  {reason:30s} {count:>5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "chip_production_data.csv",
        help="Where to write the CSV (default: data/chip_production_data.csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--days",
        type=int,
        default=DAYS_DEFAULT,
        help=(
            f"Number of production days to generate (default: {DAYS_DEFAULT}). "
            "Drift is injected only on the LAST day."
        ),
    )
    args = parser.parse_args()

    df = generate_dataset(seed=args.seed, days=args.days)
    write_csv(df, args.output)
    _print_summary(df)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
