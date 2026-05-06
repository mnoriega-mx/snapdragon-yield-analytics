"""
Load the synthetic Snapdragon production CSV into a SQLite database.

This is a one-time setup step. It reads `data/chip_production_data.csv`
(produced by `generate_data.py`) and writes `data/chip_production.db`.

The schema mirrors the column list in the project brief, plus indexes on
the columns the agent's tools query most often (timestamp, wafer_id,
test_result).

Usage:
    python data/setup_database.py
    python data/setup_database.py --csv path/to/data.csv --db path/to/out.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).parent / "chip_production_data.csv"
DEFAULT_DB = Path(__file__).parent / "chip_production.db"

TABLE_NAME = "chip_production_data"

CREATE_TABLE_SQL = f"""
CREATE TABLE {TABLE_NAME} (
    timestamp             TEXT    NOT NULL,
    wafer_id              TEXT    NOT NULL,
    chip_id               TEXT    NOT NULL PRIMARY KEY,
    soc_model             TEXT    NOT NULL,
    process_node          TEXT    NOT NULL,
    npu_tops              REAL    NOT NULL,
    npu_power_w           REAL    NOT NULL,
    cpu_freq_ghz          REAL    NOT NULL,
    memory_bandwidth_gbps REAL    NOT NULL,
    die_temp_c            REAL    NOT NULL,
    test_result           TEXT    NOT NULL CHECK (test_result IN ('PASS', 'FAIL')),
    failure_reason        TEXT
);
"""

INDEX_STATEMENTS = [
    f"CREATE INDEX idx_{TABLE_NAME}_timestamp ON {TABLE_NAME} (timestamp);",
    f"CREATE INDEX idx_{TABLE_NAME}_wafer ON {TABLE_NAME} (wafer_id);",
    f"CREATE INDEX idx_{TABLE_NAME}_result ON {TABLE_NAME} (test_result);",
]


def load_csv_into_db(csv_path: Path, db_path: Path) -> int:
    """Replace the chip_production database with a fresh copy of the CSV.

    Returns the number of rows inserted.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Run `python data/generate_data.py` first."
        )

    df = pd.read_csv(csv_path)

    # NaN -> None so SQLite stores NULL rather than the string "nan".
    df = df.where(pd.notnull(df), None)

    if db_path.exists():
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        for stmt in INDEX_STATEMENTS:
            cur.execute(stmt)
        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        conn.commit()

        # Sanity check
        (count,) = cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()
        return int(count)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    try:
        rows = load_csv_into_db(args.csv, args.db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {rows:,} rows into {args.db}")
    print(f"Table: {TABLE_NAME}")
    print("Indexes: timestamp, wafer_id, test_result")


if __name__ == "__main__":
    main()
