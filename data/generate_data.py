"""
Synthetic data generator for the Snapdragon Yield Analytics demo.

Produces a CSV containing 24 hours of fictional Snapdragon 8 Gen 5 (3nm)
production test results, with 10,000 chips spread across 100 wafers.

The dataset has a deliberate fault injected at hour 14 so the AI agent has
something concrete to find:

    Hours 00:00 to 13:59 (normal operation)
        Every chip sampled from the normal distribution.
        NPU TOPS               mean 50.5,  std 1.2
        NPU Power (W)          mean 3.20,  std 0.10
        CPU frequency (GHz)    mean 3.45,  std 0.05
        Memory bandwidth (GB/s) mean 205,   std 5
        Die temperature (C)    mean 78,    std 3
        Expected yield: ~95 percent

    Hours 14:00 to 23:59 (process drift on the NPU power domain)
        A fraction (DRIFT_AFFECTED_FRACTION, default 0.32) of chips are
        affected by an NPU-domain process excursion:
            NPU TOPS           mean 42.0,  std 2.5  (drops, more variance)
            NPU Power (W)      mean 4.5,   std 0.30 (rises sharply)
            Other metrics: identical to normal operation
        The remaining drift-hour chips sample from the normal distribution.
        Expected drift-hour yield: ~68 percent.

This mirrors how real fab process drift presents: an excursion affects a
subset of chips passing through a tool, not 100 percent of them. Among the
failed drift-hour chips, the average NPU throughput sits around 42 TOPS,
matching the demo storyline.

A chip is PASS only if every metric meets spec. failure_reason is set to
the first failed metric (in a documented priority order) so the agent can
group failures by root cause.

Usage:
    python data/generate_data.py
    python data/generate_data.py --output data/chip_production_data.csv

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
TOTAL_CHIPS = 10_000
WAFERS = 100
CHIPS_PER_WAFER = TOTAL_CHIPS // WAFERS

# Fraction of chips in drift hours that are actually affected by the
# excursion. Tuned so the drift-hour yield lands around 68 percent.
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


def generate_dataset(
    seed: int = 42,
    drift_affected_fraction: float = DRIFT_AFFECTED_FRACTION,
) -> pd.DataFrame:
    """Return a deterministic synthetic Snapdragon test dataset.

    Args:
        seed: Random seed for reproducibility.
        drift_affected_fraction: Fraction of drift-hour chips that exhibit
            the NPU excursion. Default tuned to produce ~68 percent
            drift-hour yield.
    """
    rng = np.random.default_rng(seed)

    # Evenly spaced timestamps across 24 hours.
    seconds_total = HOURS_TOTAL * 3600
    seconds_per_chip = seconds_total / TOTAL_CHIPS
    offsets = np.arange(TOTAL_CHIPS) * seconds_per_chip
    timestamps = [START_TIME + timedelta(seconds=float(s)) for s in offsets]
    hours = np.array([t.hour for t in timestamps])

    # Wafer / chip identifiers. Wafers tile across the day so each wafer
    # contains chips from a contiguous time window. This matches how a real
    # fab runs: one wafer's chips are tested back to back.
    wafer_idx = np.repeat(np.arange(WAFERS), CHIPS_PER_WAFER)
    chip_idx = np.arange(TOTAL_CHIPS)
    wafer_ids = [f"W{w:03d}" for w in wafer_idx]
    chip_ids = [f"C{c:05d}" for c in chip_idx]

    # Start by sampling every chip from the normal distribution.
    samples = _sample_profile(rng, NORMAL, TOTAL_CHIPS)

    # In drift hours, replace a fraction of chips with samples from the
    # drift distribution. We choose them with a Bernoulli mask so the
    # affected chips are spread across drift-hour wafers, mirroring an
    # excursion that hits the NPU power domain intermittently.
    drift_mask_hours = hours >= DRIFT_START_HOUR
    drift_eligible_idx = np.where(drift_mask_hours)[0]
    affected_count = int(round(len(drift_eligible_idx) * drift_affected_fraction))
    affected_idx = rng.choice(drift_eligible_idx, size=affected_count, replace=False)
    affected_idx.sort()

    drift_samples = _sample_profile(rng, DRIFT, affected_count)
    for key, arr in drift_samples.items():
        samples[key][affected_idx] = arr

    npu_tops = samples["npu_tops"]
    npu_power = samples["npu_power_w"]
    cpu_freq = samples["cpu_freq_ghz"]
    mem_bw = samples["memory_bandwidth_gbps"]
    die_temp = samples["die_temp_c"]

    # Round to realistic precision
    df = pd.DataFrame({
        "timestamp": timestamps,
        "wafer_id": wafer_ids,
        "chip_id": chip_ids,
        "soc_model": "SD8Gen5",
        "process_node": "3nm",
        "npu_tops": np.round(npu_tops, 2),
        "npu_power_w": np.round(npu_power, 3),
        "cpu_freq_ghz": np.round(cpu_freq, 3),
        "memory_bandwidth_gbps": np.round(mem_bw, 1),
        "die_temp_c": np.round(die_temp, 1),
    })

    # Decide pass/fail using the rounded values so any downstream consumer
    # of the CSV would compute the same result we did.
    failure_reasons = df.apply(_classify_failure, axis=1)
    df["test_result"] = np.where(failure_reasons.isna(), "PASS", "FAIL")
    df["failure_reason"] = failure_reasons

    return df


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Format timestamps as ISO 8601 for SQLite compatibility.
    df_out = df.copy()
    df_out["timestamp"] = df_out["timestamp"].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
    df_out.to_csv(output_path, index=False)


def _print_summary(df: pd.DataFrame) -> None:
    pass_rate_total = (df["test_result"] == "PASS").mean()
    hours = df["timestamp"].dt.hour if pd.api.types.is_datetime64_any_dtype(df["timestamp"]) else pd.to_datetime(df["timestamp"]).dt.hour
    normal_mask = hours < DRIFT_START_HOUR
    drift_mask = hours >= DRIFT_START_HOUR

    normal_yield = (df.loc[normal_mask, "test_result"] == "PASS").mean()
    drift_yield = (df.loc[drift_mask, "test_result"] == "PASS").mean()

    print(f"Generated {len(df):,} chip records.")
    print(f"  Overall yield:           {pass_rate_total:.1%}")
    print(f"  Normal hours (00 to 13): {normal_yield:.1%}  (n={int(normal_mask.sum()):,})")
    print(f"  Drift hours  (14 to 23): {drift_yield:.1%}  (n={int(drift_mask.sum()):,})")
    print()
    print("Top failure reasons:")
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
    args = parser.parse_args()

    df = generate_dataset(seed=args.seed)
    write_csv(df, args.output)
    _print_summary(df)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
