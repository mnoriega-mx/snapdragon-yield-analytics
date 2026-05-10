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

**Day 3 done.** Tool 2 (`calculate_spc_metrics`) and Tool 3 (`detect_anomalies`), both pure pandas/scipy with no API calls. Shared `_load_dataframe` helper reads windows into pandas via the same read-only connection. Tool 2 reports mean, sample std, UCL/LCL at mean +/- 3 sigma, plus per-subgroup means (by hour or wafer_id) and out-of-control flags. Tool 3 buckets the window into hourly groups, flags hours over a failure-rate threshold, and Pearson-correlates each metric's hourly mean with the hourly failure rate.

**Day 4 done.** Tool 4 (`generate_chart`) with three matplotlib templates: `spc_chart` (hourly mean line with mean and +/- 3 sigma reference lines, OOC hours highlighted), `correlation_chart` (two metrics on dual y-axes with Pearson r in the title), `failure_timeline` (one dot per failed chip, y-axis = failure_reason, color by reason). Headless `Agg` backend so charts render in tests; PNGs land in `charts/` (or a caller-supplied `output_dir`) with a microsecond-stamped filename and the tool returns the absolute path. The agent loop's `_summarize_tool_result` was extended to give compact one-line traces for SPC, anomalies, and chart calls instead of the prior `row_count=None`.

**Day 5 done.** Tool 5 (`write_summary_report`) renders a markdown yield report from `findings` (list of `{category, description, evidence}`), `root_cause_hypothesis`, and `recommendations`. `run_agent` gained an optional `on_step` callback called after each iteration; each `tool_call` entry now also records `chart_path` (for `generate_chart`) and `report` (for `write_summary_report`) so a UI can pull artifacts out without re-parsing the raw tool result. The system prompt got a new principle nudging the agent to finish substantive yield investigations with `write_summary_report`. The Streamlit UI lives at `ui/app.py`: sidebar with sample questions and project framing, question input, live `st.status` panel that streams each tool call as it happens, then the rendered layout (charts inline, report markdown, agent prose answer, full trace in a collapsible expander). Run with `streamlit run ui/app.py`.

**Day 6 done.**

- **Multi-day data layer.** `data/generate_data.py` was refactored to produce N production days (default `DAYS_DEFAULT = 7`), with the drift excursion injected only on the LAST day. Wafer ids and chip ids are unique across days (W000-W099 on day 1, W100-W199 on day 2, etc.). The bundled database (`data/chip_production.db`) now holds 70,000 rows. The agent's system prompt teaches it that "today" means the most recent day, found via `query_database(query_type='summary')`'s `last_timestamp` field. So the canonical drift day for the demo is 2026-04-07, not 2026-04-01.
- **Dashboard panel in Streamlit.** Above the question input, the app shows a "Today's production" section: KPI cards (chips tested, yield with delta vs prior 6-day average, failures with friendly labels, status pill: Alert/Watch/OK), a comparison caption, and a 24-hour yield trend line chart. Data is fetched via `query_database` with a 60-second `st.cache_data` TTL. Sample-question buttons use `on_click` callbacks (canonical Streamlit pattern), and the live trace shows friendly labels ("Generating spc chart for npu_tops") instead of raw function calls.
- **Token-budget changes.** The summary tool now returns a compact `daily_yield` rollup plus only the last day's `hourly_yield` for multi-day windows (12KB → 2.5KB JSON). The system prompt nudges the agent to scope follow-up calls to single-day windows.
- **Prompt caching.** `agent/agent.py` adds two `cache_control` breakpoints per request via `_tools_with_cache` (caches the system + tools prefix, ~3.5K tokens) and `_messages_with_cache` (caches the conversation prefix up through the latest message). Cache reads cost 10 percent and do not count toward the rate limit, so a Tier 1 key (30K input tokens/min) can comfortably handle a 4-5 iteration agent run.
- **Prompt tweaks.** Added a "data swim lane" principle (no VRMs, voltage rails, fab steps). Style block now allows `--` (double dashes); only em dashes are forbidden.
- **Response restructure.** The structured markdown from `write_summary_report` is now the canonical user-facing deliverable. The agent's final assistant text is a single short acknowledgement; there is no separate prose "cover note" or "bottom line". `write_summary_report` was renamed `bottom_line` to `root_cause_hypothesis` (matching the brief), and the section header changed from "## Bottom line" to "## Root cause hypothesis". Findings carry `{{chart:...}}` tokens inline in their `description` field where a chart visually reinforces a specific finding; the Streamlit UI's `_render_markdown_with_charts` expands those tokens into inline charts. The UI no longer echoes the question or renders the prose answer when a report exists.
- **`load_dotenv(override=True)`** so an empty shell-level `ANTHROPIC_API_KEY` cannot mask the real key in `.env`.
- **Basic logging.** New `agent/logging_setup.py` exposes `setup_file_logging(log_dir)` which is idempotent and adds a per-run FileHandler at `logs/agent_YYYYMMDD_HHMMSS.log`. The library never installs handlers itself: `agent/run.py` (CLI) and `ui/app.py` (Streamlit) call it once at startup; tests do not, so the suite never writes log files. `tools.execute_tool` logs each call with name, duration, args, and status (`ok`, `error`, `crash`, `unknown`). `agent.run_agent` logs run start, every iteration with token usage and tool names, and run end (or `cap_reached=true` when the iteration cap fires). `logs/` is gitignored.
- **End-to-end agent test.** `tests/test_agent_e2e.py` drives `run_agent` against a `FakeAnthropicClient` that returns scripted Anthropic responses. Coverage: full yield-investigation happy path (4 tool calls then end_turn), iteration safety cap, tool errors flowing back to Claude as a `tool_result` block, the `on_step` callback firing once per iteration in order, prompt-cache breakpoints landing on the last tool and last message block, token usage extraction from `response.usage`, and unknown-tool-name handling. Real-API calls do not happen here.
- **Formal scenario validation.** `scripts/validate_scenarios.py` runs the three brief scenarios (Normal operation, Anomaly investigation, Specific lookup) against the live Anthropic API. Each scenario carries soft expectations (concept presence, tools called); the script prints PASS/FAIL on stdout and writes a markdown summary to `docs/scenario_validation.md`. Invoke before recording the demo: `python scripts/validate_scenarios.py`. Exit code is 0 only when every expectation passes.

Current test status: 74 tests passing (12 data, 55 tools, 7 e2e).

**Day 7 in progress: deployment prep done.**

- **Stderr log handler.** `setup_file_logging` now attaches both a FileHandler (per-run file in `logs/`) and a StreamHandler on `sys.stderr`. The stderr stream is what gives a hosted deployment a useful live log view: every tool call shows up in Streamlit Community Cloud's manage-page log viewer in real time, since the per-run file there is ephemeral.
- **`data/chip_production.db` is committed.** Previously gitignored; now eligible to be tracked so a fresh clone (which is what the hosting platform does) has data on first boot. The DB is ~12 MB and deterministic from `seed=42`; rebuild it locally and recommit if the generator changes. `data/*.db-journal` and `data/chip_production_data.csv` stay ignored.
- **Hosting.** App is deployed on Streamlit Community Cloud. `ANTHROPIC_API_KEY` is set as a Streamlit secret in the app settings; Streamlit Cloud injects top-level secrets as environment variables so the existing `os.getenv` lookup in `agent/agent.py` finds the key without code changes. The live URL is shared with recruiters directly rather than published in the README, so the API budget stays predictable.
- **`docs/deploy.md` removed.** The README and the platform's own UI cover the small amount of deployment know-how needed; a separate guide added churn without value.

Day 7 still pending: README polish, architecture diagram, demo screenshots, demo video, LICENSE.

## Critical project rules (do not violate)

These come from section 13 of the brief. They are non-negotiable.

1. **No em dashes (`—`) anywhere.** Not in code comments, not in user-visible text, not in the README, not in the agent's system prompt. Use commas, parentheses, semicolons, or `--` instead. Double dashes (`--`) are fine and can be used freely. (Earlier copies of this file said "no double dashes" too; that was a misnaming. The real rule is em dashes only.)
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
│   └── chip_production.db        (committed, ~12 MB, regenerable from seed=42)
├── agent/
│   ├── __init__.py
│   ├── prompts.py              System prompt for the agent
│   ├── tools.py                Tool catalog (5 of 5 tools implemented)
│   ├── agent.py                Claude API tool-use loop
│   ├── logging_setup.py        Per-run file logging helper
│   └── run.py                  CLI runner: python -m agent.run "your question"
├── ui/
│   └── app.py                  Streamlit UI: streamlit run ui/app.py
├── scripts/
│   └── validate_scenarios.py   Run the 3 brief scenarios; writes docs/scenario_validation.md
├── tests/
│   ├── conftest.py             Shared fixtures (chip_db builds a fresh test DB)
│   ├── test_data_generation.py
│   ├── test_tools.py
│   └── test_agent_e2e.py       Agent loop tests against a FakeAnthropicClient
├── docs/
│   ├── project_brief.md        The full brief
│   ├── architecture.png        (TBD: still pending Day 7)
│   ├── scenario_validation.md  (generated by scripts/validate_scenarios.py)
│   └── demo_screenshots/       (TBD: still pending Day 7)
├── charts/                     (gitignored, populated at runtime)
├── logs/                       (gitignored, one log file per run)
├── requirements.txt
├── setup.sh                    One-command setup
├── .env.example                Template; copy to .env, then add real key
├── .gitignore
└── README.md                   Public-facing project explanation (per brief section 11, minus Why/Limitations/About)
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

Launch the Streamlit UI:
```
streamlit run ui/app.py
```

Run the formal scenario validation (hits the live API, writes `docs/scenario_validation.md`):
```
python scripts/validate_scenarios.py
python scripts/validate_scenarios.py --scenario "Anomaly investigation"
```

## Mauricio's preferences

- Avoids em dashes (`—`) in writing. Double dashes (`--`) are fine.
- Prefers prose over heavy bullet/header formatting in chat responses.
- Is not a deep semiconductor expert; he is a data/AI person making a portfolio piece. Don't pretend otherwise in the README or any user-facing copy. Stay in the data and AI swim lane.
- Project venv lives at `./venv`, built from homebrew Python 3.12 (`/opt/homebrew/bin/python3.12`). Activate with `source venv/bin/activate`, or invoke directly via `./venv/bin/python` and `./venv/bin/pytest`. Anaconda Python 3.9 is still on the machine but is not the venv base.

## When you start a new session

Read `docs/project_brief.md` first (it is long but worth it). Then check what day we are on by skimming `tests/` and the existing files in `agent/`. Then ask Mauricio what he wants to tackle.
