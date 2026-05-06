"""
Shared pytest fixtures for the Snapdragon Yield Analytics test suite.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Make the project root importable for tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def chip_db(tmp_path_factory) -> Path:
    """Build a fresh SQLite database from the deterministic seed and yield its path.

    Building the DB once per session keeps tests fast. We always rebuild
    from `data.generate_data.generate_dataset(seed=42)` so the tests do
    not rely on any artifact the user happens to have on disk.
    """
    from data.generate_data import generate_dataset
    from data.setup_database import (
        CREATE_TABLE_SQL,
        INDEX_STATEMENTS,
        TABLE_NAME,
    )

    db_path = tmp_path_factory.mktemp("db") / "chip_production.db"
    df = generate_dataset(seed=42)
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.where(df.notnull(), None)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        for stmt in INDEX_STATEMENTS:
            cur.execute(stmt)
        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()

    return db_path
