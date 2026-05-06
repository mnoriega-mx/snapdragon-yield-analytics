# Snapdragon Yield Analytics: Working Context for Claude

This file is auto-loaded by Claude Code when it opens this folder. It exists so a fresh Claude session can pick up where the last one left off without Mauricio having to re-explain everything.

## What this project is

A 7-day portfolio demo for Mauricio's application to a Qualcomm summer internship: **Summer Intern, Machine Learning FSDO System Analyst (Job ID 3086988)**.

The deliverable is a Streamlit web app where a yield engineer asks plain-language questions about Snapdragon SoC production data, and a Claude agent orchestrates 5 predefined tools to deliver root cause analysis in under 30 seconds.

The full brief lives at `docs/project_brief.md`. Read that before doing anything substantive. Sections worth knowing by name:
- Section 5 spells out the synthetic dataset's distribution parameters
- Section 7 spells out each of the 5 tools' input/output schemas
- Section 8 is the day-by-day build plan
- Section 13 lists the non-negotiable constraints

## Where we are right now

**Day 1 done.** Synthetic dataset, SQLite database, data-generation tests.

**Day 2 done.** System prompt, the first tool (`query_database`), the Claude API agent loop, a CLI runner, and unit tests for the tool.

**Day 3 next.** Implement Tool 2 (`calculate_spc_metrics`) and Tool 3 (`detect_anomalies`). Both are pure-Python over pandas/scipy, no API calls.

Current test status: 26 tests passing (11 data, 15 tools).

## Critical project rules (do not violate)

These come from section 13 of the brief. They are non-negotiable.

1. **No double dashes (`--`) or em dashes (`—`) anywhere.** Not in code comments, not in user-visible text, not in the README, not in the agent's system prompt. This is a personal style rule for Mauricio. Use commas, parentheses, or semicolons instead.
2. **The product is "Snapdragon SoC", not "Hexagon NPU".** The Hexagon NPU is one subsystem on the chip. Never imply Qualcomm makes a standalone NPU or runs a "Hexagon NPU factory". The framing is always "Snapdragon production with focus on Hexagon NPU testing".
3. **Don't invent absurd specs.** Section 15 of the brief lists the canonical numbers: 50 TOPS target, 48 TOPS pass, 3.2 W target, 3.5 W pass, 3nm process, 10,000 chips/day. Don't make up numbers that contradict these.
4. **The agent uses predefined tools only, no free-form code execution.** This matches the FSDO job description's "predefined code paths" language and is one of the things the demo is meant to showcase.
5. **All synthetic data must be labeled as synthetic in any user-facing copy.** No real Qualcomm data is used.
6. **Claude is the LLM of choice.** Not GPT, not Gemini. The JD lists Claude in preferred quals.

## Tech stack

- Python 3.9+ on Mauricio's Mac (anaconda). Brief recommends 3.11+ but code uses `from __future__ import annotations` so it runs fine on 3.9.
- pandas, numpy, scipy for data and stats
- matplotlib, seaborn for charts (Day 4)
- anthropic SDK for the Claude API
- streamlit for the UI (Day 5)
- pytest for tests
- SQLite for the production database

## Repo layout

```
.
├── data/
│   ├── generate_data.py        Synthetic test data generator (deterministic, seed=42)
│   ├── setup_database.py       CSV -> SQLite loader
│   ├── chip_production_data.csv  (gitignored)
│   └── chip_production.db        (gitignored)
├── agent/
│   ├── __init__.py
│   ├── prompts.py              System prompt for the agent
│   ├── tools.py                Tool catalog (1 of 5 tools implemented so far)
│   ├── agent.py                Claude API tool-use loop
│   └── run.py                  CLI runner: python -m agent.run "your question"
├── ui/                         (empty for now, Streamlit app on Day 5)
├── tests/
│   ├── conftest.py             Shared fixtures (chip_db builds a fresh test DB)
│   ├── test_data_generation.py
│   └── test_tools.py
├── docs/
│   ├── project_brief.md        The full brief
│   ├── architecture.png        (TBD on Day 7)
│   └── demo_screenshots/       (TBD on Day 7)
├── charts/                     (gitignored, populated at runtime)
├── requirements.txt
├── setup.sh                    One-command setup
├── .env.example                Template; copy to .env, then add real key
├── .gitignore
└── README.md                   (TBD on Day 7)
```

## Important data-layer details

The brief's section 5 says drift hours should have NPU TOPS mean 42, std 2.5, and yield ~68 percent. Those numbers are mathematically inconsistent: a normal distribution with mean 42 and std 2.5 fails the 48 TOPS threshold ~99 percent of the time, not 32 percent.

The reconciliation: in `data/generate_data.py` the constant `DRIFT_AFFECTED_FRACTION = 0.32` controls what fraction of drift-hour chips are actually affected by the excursion. The other 68 percent sample from the normal distribution. This produces the 68 percent yield target while keeping the "failed chips average 42 TOPS" narrative true (it is the average among the failing subset, not the global drift-hour average). Documented in the generator's docstring. Don't break this without good reason.

## Agent design choices

- The system prompt in `agent/prompts.py` tells Claude to always reach for a tool before stating numeric facts. This is what the JD calls "predefined code paths."
- `query_database` accepts only four `query_type` values (`summary`, `date_range`, `failed_only`, `wafer_range`). Claude never gets free-form SQL.
- The connection is opened in read-only mode (`mode=ro`) as defense in depth.
- Returned rows are capped at 2000 (`MAX_ROWS_RETURNED`) to protect Claude's context window.
- Default model: `claude-sonnet-4-6`. Override via `ANTHROPIC_MODEL` in `.env` if needed. Sonnet is the right pick for this demo (good tool-use, reasonable cost, fits the FSDO JD's mention of Claude).

## How to run things

Quick smoke test the data layer:
```
python data/generate_data.py
python data/setup_database.py
```

Run the full test suite:
```
python -m pytest -v
```

Ask the agent a question (requires `.env` with `ANTHROPIC_API_KEY`):
```
python -m agent.run "How many chips were produced today?"
python -m agent.run --trace "Why did yield drop today?"
```

## Mauricio's preferences

- Avoids double dashes in writing (the `--` style). Use commas or other punctuation.
- Prefers prose over heavy bullet/header formatting in chat responses.
- Is not a deep semiconductor expert; he is a data/AI person making a portfolio piece. Don't pretend otherwise in the README or any user-facing copy. Stay in the data and AI swim lane.
- Currently running anaconda Python 3.9, no project venv yet. Code is compatible with 3.9 but a venv is recommended.

## When you start a new session

Read `docs/project_brief.md` first (it is long but worth it). Then check what day we are on by skimming `tests/` and the existing files in `agent/`. Then ask Mauricio what he wants to tackle.
