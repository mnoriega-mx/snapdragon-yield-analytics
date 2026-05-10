"""
Per-run logging for the Snapdragon Yield Analytics agent.

The agent loop and the tool dispatcher emit log records on the
`snapdragon_agent` logger tree, but the library never installs handlers
itself. Entry points (`agent.run`, `ui/app.py`, the scenario harness)
call `setup_file_logging` once at startup, which attaches two handlers:

    1. A FileHandler at logs/agent_YYYYMMDD_HHMMSS.log, useful locally
       for inspecting a run after the fact.
    2. A StreamHandler on stderr, captured by hosted platforms like
       Streamlit Community Cloud so the manage-page log viewer shows
       every tool call as it happens.

Tests do not call this function, so test runs stay silent and do not
pollute the logs directory or the test output.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "snapdragon_agent"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LEVEL = logging.INFO

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger() -> logging.Logger:
    """Return the project's named logger."""
    return logging.getLogger(LOGGER_NAME)


def setup_file_logging(
    log_dir: Path | str | None = None,
    level: int = DEFAULT_LEVEL,
) -> Path:
    """Attach per-run file and stderr handlers to the project logger.

    Returns the path of the log file. Idempotent: if a FileHandler is
    already attached, returns its path without adding a second copy of
    either handler.

    The stderr stream handler is what gives a hosted Streamlit Cloud
    deployment a useful "live console" view: every tool call shows up
    in the manage-page log viewer in real time.
    """
    logger = get_logger()
    logger.setLevel(level)

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)

    out_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"agent_{stamp}.log"

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Keep records out of the root logger so Streamlit's own log
    # plumbing does not double-print them.
    logger.propagate = False

    logger.info("logging initialized at %s", path)
    return path
