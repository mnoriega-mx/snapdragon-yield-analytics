# AI-Powered Yield Analytics for Snapdragon SoC Production

## Project Brief for Claude Cowork

---

## 1. Context & Purpose

This is a portfolio demo project built to support an internship application to Qualcomm. The target role is **Summer Intern, Machine Learning FSDO System Analyst (Job ID 3086988)**, which sits in Qualcomm's Fab Systems and Data Operations team. FSDO manages the data infrastructure that validates chips after fabrication, and the team is building AI agents that automate routine analysis workflows for yield engineers.

The role's job description emphasizes four things:

1. **Workflow design**: AI agents that orchestrate tasks through predefined code paths.
2. **Integration with existing systems**: Seamless connection to data infrastructure (databases, ETL, retrieval).
3. **End-to-end analysis**: Preprocessing → insights → visualizations, all autonomously.
4. **Operational validation**: Test, evaluate, and optimize for stability and scalability.

This project demonstrates all four capabilities in a single end-to-end demo, framed in Qualcomm's actual product domain (Snapdragon SoCs with Hexagon NPU subsystems).

The audience for the GitHub repo and demo video is the Qualcomm recruiter and hiring manager. The goal is to make them think: "This person already gets what we do. We should interview them."

---

## 2. The Problem the Demo Solves

### The universal manufacturing problem

Imagine a factory that produces 10,000 units per day. Every unit is tested before shipping. Sometimes yield drops unexpectedly: yesterday 95% of units passed, today only 70%. Something broke. The yield engineer needs to find the root cause fast, because every minute of bad production is lost revenue.

The traditional workflow:

1. Engineer opens a database tool, exports today's production data.
2. Loads it into Excel or a notebook.
3. Filters by machine, by time, by failure type.
4. Builds charts to visualize patterns.
5. Cross-references sensor data to find the smoking gun.
6. Writes a summary report.
7. Total time: 2 hours, often more.

The AI-augmented workflow:

1. Engineer asks: "Why did yield drop today?"
2. AI agent queries the database, runs statistics, detects anomalies, generates charts, writes a summary, and recommends next steps.
3. Total time: 30 seconds.

This demo builds the second workflow.

### The Qualcomm-specific framing

Qualcomm doesn't make widgets, they make Snapdragon SoCs (System-on-Chip processors used in Android flagship phones). Each Snapdragon contains multiple subsystems: Kryo CPU, Adreno GPU, Hexagon NPU (AI accelerator), Spectra ISP, and a Snapdragon X-series modem. They are all fabricated together on a single chip at foundry partners like TSMC or Samsung.

After fabrication, every chip is tested. One critical test is **AI performance binning** for the Hexagon NPU, which measures:

- **Inference throughput** (TOPS, Tera Operations Per Second)
- **Power efficiency** (TOPS per Watt)
- **Memory bandwidth** between the NPU and other subsystems
- **Thermal characteristics** under sustained AI workloads

Chips that fail target specs cause yield drops, which translate directly into revenue loss. This demo simulates that exact scenario for a fictional Snapdragon 8 Gen 5 production line, with the agent identifying the root cause of NPU performance failures.

### Important framing note

The product is **Snapdragon SoCs**, the focus is **Hexagon NPU validation**. Qualcomm does not sell standalone NPUs, so the demo should never imply "Hexagon NPU factory." Always frame as "Snapdragon production, with focus on Hexagon NPU testing."

---

## 3. What the Demo Actually Does (User's Perspective)

### Scenario

A yield engineer at a Qualcomm-affiliated fab notices today's yield is lower than expected. They open the AI Analytics tool and ask a question in natural language.

### User interaction

**Step 1**: Engineer opens the Streamlit web app.

**Step 2**: Engineer types: "Why did yield drop this afternoon?"

**Step 3**: Behind the scenes, the agent:
- Calls `query_database(filter="today")` to retrieve production data
- Calls `calculate_spc_metrics()` to compute control limits and detect statistical anomalies
- Calls `detect_anomalies()` to find correlated patterns across NPU TOPS, power, and bandwidth
- Calls `generate_chart()` to plot the temperature, performance, and failure timeline
- Calls `write_summary_report()` to produce a human-readable output

**Step 4**: Engineer sees:
- A text summary explaining what happened
- A control chart showing NPU performance over time, with anomalies marked in red
- A correlation chart linking NPU performance with power consumption
- A root cause hypothesis
- A recommended next action

Total time from question to answer: under 30 seconds.

### Sample output (what the agent would produce)

```
Summary:
Yield drop detected in Snapdragon 8 Gen 5 production. Chips from
Wafer Lot W050 to W055 are failing Hexagon NPU performance binning,
with average NPU throughput at 42 TOPS versus the 50 TOPS target
(16 percent below spec).

Root Cause:
Hexagon NPU subsystem underperforming due to process variation
during lithography. Correlated with elevated power consumption
(4.5W average versus 3.2W target), suggesting voltage and frequency
scaling issue in the NPU power domain.

Impact:
- 6,000 chips affected
- Yield dropped from 95 percent to 68 percent
- Affected chips may be eligible for downbin to mid-tier Snapdragon 7+
  Gen 4 if CPU and GPU subsystems still meet spec

Recommendation:
1. Inspect lithography tool calibration for the NPU block
2. Review voltage regulator settings for Hexagon power domain
3. Run secondary validation on affected lots for downbin eligibility
```

---

## 4. Tech Stack

The whole stack is intentionally lightweight so the demo runs anywhere with minimal setup.

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Universal, fits the data analyst tooling |
| Database | SQLite | No server setup, file-based, real SQL |
| Data manipulation | pandas, numpy | Standard for tabular analysis |
| Statistics | scipy.stats | For SPC control limits |
| Charts | matplotlib, seaborn | Standard, embeds easily in Streamlit |
| AI orchestration | Anthropic Claude API with tool use | Matches Qualcomm JD's "GenAI experience with Claude" |
| UI | Streamlit | Fast prototyping, professional-looking output |
| Environment | Python venv | Reproducibility |
| Version control | Git, GitHub | Public repo for the recruiter to review |

No cloud deployment is required for the demo. A local run is enough. If extra polish is wanted at the end, deploy the Streamlit app to Streamlit Cloud for a live link.

---

## 5. Synthetic Dataset Specification

The dataset is fully synthetic. The goal is to simulate realistic Snapdragon SoC test data with a clearly defined "problem hour" so the agent has something concrete to find.

### Schema (one main table)

```
Table: chip_production_data

| Column                  | Type       | Description                                      |
|-------------------------|------------|--------------------------------------------------|
| timestamp               | DATETIME   | When the chip was tested                         |
| wafer_id                | TEXT       | Wafer identifier (e.g., "W050")                  |
| chip_id                 | TEXT       | Unique chip identifier (e.g., "C05000")          |
| soc_model               | TEXT       | Always "SD8Gen5" for this demo                   |
| process_node            | TEXT       | "3nm" (matches Snapdragon 8 Elite generation)    |
| npu_tops                | REAL       | Hexagon NPU throughput in TOPS                   |
| npu_power_w             | REAL       | NPU power consumption in Watts                   |
| cpu_freq_ghz            | REAL       | Kryo CPU max frequency in GHz                    |
| memory_bandwidth_gbps   | REAL       | NPU-to-memory bandwidth in GB/s                  |
| die_temp_c              | REAL       | Die temperature during sustained NPU workload    |
| test_result             | TEXT       | "PASS" or "FAIL"                                 |
| failure_reason          | TEXT       | Why it failed, or NULL if passed                 |
```

### Realistic spec targets (used for pass/fail criteria)

These numbers are grounded in publicly known Snapdragon specs, projected slightly forward to make the fictional "8 Gen 5" plausible.

| Metric | Target | Pass Threshold | Notes |
|---|---|---|---|
| NPU TOPS | 50 | ≥ 48 | Snapdragon 8 Elite hits 45 TOPS, so 50 is a reasonable next gen |
| NPU Power | 3.2W | ≤ 3.5W | Educated estimate, real numbers not public |
| CPU Freq | 3.4 GHz | ≥ 3.3 GHz | In line with current Snapdragon 8 Elite |
| Memory Bandwidth | 200 GB/s | ≥ 190 GB/s | LPDDR5X-class numbers |
| Die Temp | < 95°C | < 95°C | Standard thermal envelope |

A chip is `PASS` only if ALL thresholds are met. Otherwise `FAIL`, with `failure_reason` set to the first failed metric.

### Data volume

- **Time range**: 24 hours of production (e.g., 2026-04-01 00:00 to 23:59)
- **Volume**: 10,000 chips total, 100 wafers, 100 chips per wafer
- **Distribution**: Uniformly distributed across the 24 hours

### The injected problem

For the demo to have a clear story, we inject a known fault:

- **Hours 00:00 to 13:59 (normal operation)**:
  - NPU TOPS: normal distribution, mean 50.5, std dev 1.2
  - NPU Power: mean 3.2W, std dev 0.1W
  - Other metrics: normal
  - Expected yield: ~95 percent

- **Hours 14:00 to 23:59 (process drift)**:
  - NPU TOPS: mean drops to 42, std dev increases to 2.5
  - NPU Power: mean rises to 4.5W, std dev 0.3W
  - Other metrics: still normal (this is important, the agent should isolate the NPU as the root cause)
  - Expected yield: ~68 percent

This gives the agent a concrete pattern to find: yield drops at 14:00, correlated with NPU performance degradation and power increase, while CPU/memory/thermal stay normal. The agent's job is to surface this without being told about it.

### Generation script

A `data/generate_data.py` file produces this CSV using numpy's normal distribution. Includes a fixed random seed so results are reproducible.

---

## 6. Architecture Overview

### High-level flow

```
User question (natural language)
        │
        ▼
   Streamlit UI
        │
        ▼
  Claude API agent loop  ◄──────────┐
        │                            │
        ▼                            │
  Tool selection                     │
        │                            │
        ▼                            │
  ┌─────┴────────────────────┐       │
  ▼     ▼     ▼     ▼     ▼  ▼       │
query  spc  detect  chart  report   │
  │     │     │     │     │  │       │
  ▼     ▼     ▼     ▼     ▼  ▼       │
SQLite  pandas  matplotlib  text     │
  │     │     │     │     │  │       │
  └─────┴─────┴─────┴─────┴──┘       │
        │                            │
        ▼                            │
  Tool results sent back to agent ───┘
        │
        ▼
  Final summary + chart shown to user
```

### Components

1. **Frontend**: A single-page Streamlit app with a question input, a "conversation log" panel showing which tools were called, and a results panel.

2. **Agent loop**: A Python function that sends the user's question to Claude, handles any `tool_use` responses, executes tools locally, sends results back, and continues until Claude returns a final text response.

3. **Tools**: Five Python functions exposed to Claude as a tool catalog. Each is documented and validated.

4. **Data layer**: SQLite database queried via parameterized SQL.

5. **Chart layer**: matplotlib functions that save PNG files to disk and return the file paths.

---

## 7. Tool Specifications

These are the five tools exposed to Claude. Each one is a Python function with a tightly scoped responsibility. The agent decides which to call and in what order.

### Tool 1: `query_database`

**Description**: Run a parameterized query against the production database. Used by the agent to fetch raw data for analysis.

**Input schema**:
```json
{
  "query_type": "string (one of: 'date_range', 'wafer_range', 'failed_only', 'summary')",
  "start_time": "string (ISO timestamp, optional)",
  "end_time": "string (ISO timestamp, optional)",
  "wafer_ids": "list of strings (optional)"
}
```

**Output**: JSON-serialized DataFrame with the matching rows.

**Why constrained**: The agent does not get raw SQL access for safety. It picks from predefined query types, which matches the JD's "predefined code paths" language.

### Tool 2: `calculate_spc_metrics`

**Description**: Calculate Statistical Process Control metrics for a given metric over a time window.

**Input schema**:
```json
{
  "metric": "string (one of: 'npu_tops', 'npu_power_w', 'cpu_freq_ghz', 'memory_bandwidth_gbps', 'die_temp_c')",
  "start_time": "string (ISO timestamp)",
  "end_time": "string (ISO timestamp)",
  "group_by": "string (one of: 'hour', 'wafer_id')"
}
```

**Output**: JSON with mean, standard deviation, upper and lower control limits (mean ± 3σ), and a list of out-of-control points.

### Tool 3: `detect_anomalies`

**Description**: Find time windows where the failure rate exceeded a threshold and identify which metric correlates most strongly with the failures.

**Input schema**:
```json
{
  "start_time": "string",
  "end_time": "string",
  "failure_rate_threshold": "number (default 0.10, meaning 10 percent)"
}
```

**Output**: JSON with a list of anomalous time windows, the failure rate in each, and Pearson correlation coefficients between failure rate and each test metric.

### Tool 4: `generate_chart`

**Description**: Produce a matplotlib chart visualizing one or two metrics over time, with control limits and failure markers.

**Input schema**:
```json
{
  "chart_type": "string (one of: 'spc_chart', 'correlation_chart', 'failure_timeline')",
  "primary_metric": "string",
  "secondary_metric": "string (optional)",
  "start_time": "string",
  "end_time": "string"
}
```

**Output**: Path to the saved PNG file.

### Tool 5: `write_summary_report`

**Description**: Generate a structured markdown report combining the findings.

**Input schema**:
```json
{
  "findings": "list of objects (each with 'category', 'description', 'evidence')",
  "root_cause_hypothesis": "string",
  "recommendations": "list of strings"
}
```

**Output**: Markdown text.

---

## 8. Build Sequence (7 Days)

This is the day-by-day plan. Each day produces something runnable, so the project is never broken.

### Day 1: Foundation

- Set up the project folder structure
- Create Python venv, install dependencies (`anthropic`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`, `python-dotenv`, `scipy`)
- Write `data/generate_data.py` to produce the synthetic CSV
- Run the generator, verify the dataset has the expected pattern (sanity check: hours 0 to 13 should have ~95 percent yield, hours 14 to 23 should have ~68 percent yield)
- Load the CSV into SQLite via a one-time `setup_database.py` script

**Deliverable**: A SQLite database with 10,000 chip records.

### Day 2: Core agent + first tool

- Set up the Anthropic API key in a `.env` file
- Write `agent.py` with the basic Claude API call loop
- Implement Tool 1 (`query_database`) and wire it to the agent
- Test manually: ask Claude "How many chips were produced today?", verify it calls the tool correctly

**Deliverable**: An agent that can answer simple data questions via one tool.

### Day 3: Statistical tools

- Implement Tool 2 (`calculate_spc_metrics`) using scipy and pandas
- Implement Tool 3 (`detect_anomalies`) with correlation analysis
- Test: ask "Are there anomalies in NPU performance today?", verify the agent identifies the 14:00 onwards drop

**Deliverable**: An agent that can detect statistical anomalies.

### Day 4: Visualization tools

- Implement Tool 4 (`generate_chart`)
- Build three chart templates:
  - SPC chart (single metric over time, with control limits, anomalies in red)
  - Correlation chart (two metrics on dual y-axes)
  - Failure timeline (scatter of failures over time, colored by failure reason)
- Save outputs to a `charts/` folder

**Deliverable**: Charts produced on demand by the agent.

### Day 5: Reporting + Streamlit UI

- Implement Tool 5 (`write_summary_report`)
- Build the Streamlit app: question input, conversation log, results panel
- Wire the agent loop into Streamlit so each tool call is shown in the UI in real time
- Display the generated chart inline in the results panel

**Deliverable**: A working web UI where a user can ask the question and see the agent's full reasoning and output.

### Day 6: Validation and polish

- Run three end-to-end test scenarios:
  - Normal operation question ("How is yield today?")
  - Anomaly investigation ("Why did yield drop?")
  - Specific lookup ("Show me Wafer W050's performance")
- For each, manually verify the agent's reasoning matches expectations
- Fix any tool-call loops, hallucinated metric names, or incorrect chart selections
- Add input validation so the agent fails gracefully on bad input
- Add basic logging so every tool call is recorded with timestamp and duration

**Deliverable**: A reliable, demo-ready app.

### Day 7: Documentation and release

- Write the README (see section 11 below for what it must contain)
- Add a `requirements.txt` and a `setup.sh` script for one-command setup
- Record a 2 to 3 minute demo video walking through the three test scenarios
- Push to GitHub as a public repo
- Optional: deploy to Streamlit Cloud for a live link
- Add a `LICENSE` file (MIT is fine)

**Deliverable**: Public GitHub repo + demo video, ready to share with the recruiter.

---

## 9. Folder Structure

```
snapdragon-yield-analytics/
│
├── README.md
├── LICENSE
├── requirements.txt
├── setup.sh
├── .env.example
├── .gitignore
│
├── data/
│   ├── generate_data.py
│   ├── setup_database.py
│   └── chip_production.db          (gitignored, generated locally)
│
├── agent/
│   ├── __init__.py
│   ├── agent.py                    (main loop)
│   ├── tools.py                    (the 5 tools)
│   └── prompts.py                  (system prompt for the agent)
│
├── charts/                         (gitignored, generated at runtime)
│
├── ui/
│   └── app.py                      (Streamlit entry point)
│
├── tests/
│   ├── test_data_generation.py
│   ├── test_tools.py
│   └── test_agent_e2e.py
│
└── docs/
    ├── architecture.png            (the diagram from section 6)
    └── demo_screenshots/
        ├── 01_question.png
        ├── 02_agent_reasoning.png
        └── 03_results_with_chart.png
```

---

## 10. Testing Strategy

The demo is small, but tests matter for credibility. The recruiter or hiring manager who clones the repo should be able to run `pytest` and see green checks.

### Test categories

**Data generation tests** (`test_data_generation.py`):
- The synthetic dataset has exactly 10,000 rows
- All required columns are present
- The "problem hours" (14:00 to 23:59) have a yield between 60 and 75 percent
- The "normal hours" (00:00 to 13:59) have a yield above 90 percent
- Random seed produces identical output across runs

**Tool unit tests** (`test_tools.py`):
- `query_database` returns the correct number of rows for a given time window
- `calculate_spc_metrics` produces upper and lower limits at exactly mean ± 3σ
- `detect_anomalies` correctly flags the 14:00 to 23:59 window as anomalous
- `generate_chart` produces a non-empty PNG file at the expected path
- `write_summary_report` returns valid markdown

**End-to-end agent test** (`test_agent_e2e.py`):
- Given the question "Why did yield drop today?", the agent calls at least 3 tools and produces a final summary that mentions "NPU" and "14:00" or "afternoon"
- Note: this test is non-deterministic due to LLM output, so it asserts on the presence of key concepts, not exact strings

---

## 11. README Requirements

The README is what the recruiter will actually read. It must include the following sections, in this order, with this content depth.

### Section: Title and one-liner

> # AI-Powered Yield Analytics for Snapdragon SoC Production
>
> An AI agent that automates yield root cause analysis for Snapdragon chip manufacturing, focusing on Hexagon NPU performance validation. Built with Claude, Python, and Streamlit.

### Section: Problem Statement

A 3 to 4 paragraph explanation of why yield engineers need this tool, framed in Qualcomm's domain. Reuse the language from section 2 of this brief.

### Section: Demo

A direct link to the 2 to 3 minute video, plus 3 to 5 screenshots showing the agent in action.

### Section: Architecture

The diagram from section 6, plus a short description of the agent loop and the five tools.

### Section: How It Works (Step by Step)

A walkthrough of one example run, showing:
1. The user's question
2. The tool calls the agent makes (in order)
3. What each tool returns
4. The final summary the agent produces

### Section: Why This Project

A short statement of intent: "I built this as a demo for a Qualcomm internship application. The goal was to prove I could build the kind of structured, tool-using AI agent the FSDO team is looking for, in a domain (Snapdragon production, Hexagon NPU validation) relevant to Qualcomm."

### Section: Tech Stack

Bullet list of technologies used.

### Section: How to Run It

Step-by-step setup instructions:
1. Clone the repo
2. Create venv, activate it
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env`, add Claude API key
5. `python data/generate_data.py`
6. `python data/setup_database.py`
7. `streamlit run ui/app.py`

### Section: Limitations and Future Work

Honest list of what the demo doesn't do:
- Synthetic data only (no real fab data, obviously)
- Single product (SD8Gen5) and single test type (NPU)
- No multi-user support, no auth, no deployment hardening
- The agent's reasoning is non-deterministic, so identical questions can produce slightly different outputs

Future ideas:
- Predictive SPC: flag drift before it becomes a yield drop
- Multi-product: extend to GPU and CPU subsystem testing
- Integration with real Power BI dashboards
- Slack or Teams bot interface

### Section: About

One paragraph: who built it, why, link to portfolio website (https://mauricio-noriega.vercel.app/) and LinkedIn.

---

## 12. The Cover Letter Hook (How This Project Connects to the Job)

When mentioning this project in the application, use this kind of phrasing:

> "I built a demo AI agent that analyzes Snapdragon production data, focusing on Hexagon NPU yield optimization. The agent automates root cause analysis for NPU performance failures, reducing investigation time from hours to seconds. The architecture follows a structured tool-calling pattern (5 predefined tools the agent orchestrates) which mirrors the workflow design described in the FSDO role's project brief."

In an interview, the elevator pitch is:

> "I picked Snapdragon production as the domain because it's what Qualcomm actually makes, and I wanted my demo to feel relevant rather than generic. The analytical logic (SPC, anomaly detection, correlation analysis) would work for any manufacturing process, but framing it around Hexagon NPU TOPS and power binning gave me a chance to engage with Qualcomm's product vocabulary. I'm not pretending to be a chip designer, my background is in data analytics and AI orchestration. The point of the demo is to show I can build the kind of structured, tool-using agent that accelerates analyst workflows, which is exactly what the JD asks for."

---

## 13. Critical Constraints and Reminders

These are non-negotiable while building the project:

- **No em dashes (`—`) anywhere in writing**: README, comments, summary outputs, cover letter language. Double dashes (`--`) are fine and can be used freely. This is a personal style rule.
- **Snapdragon, not standalone Hexagon**: the product is the SoC, the focus is the NPU subsystem. Never imply "Hexagon NPU factory."
- **No invented chip specs that contradict reality**: 50 TOPS for SD8Gen5 is plausible (45 today, modest projection forward). Don't invent 200 TOPS or other absurd numbers.
- **The agent must use predefined tools, not free-form code execution**: this matches the JD's "predefined code paths" language and the actual FSDO requirement.
- **No claims of deep semiconductor expertise in the README or interview**: stay in the data/AI swim lane, acknowledge fab engineers are the domain experts.
- **All synthetic data must be clearly labeled as synthetic**: the README explicitly states the data is generated, no real Qualcomm data is used.
- **Claude (not GPT, not Gemini) is the LLM of choice**: the JD lists Claude in preferred quals, and using it shows alignment with the team's stack.

---

## 14. Definition of Done

The project is complete when all of the following are true:

- [ ] GitHub repo is public at github.com/mnoriega-mx/snapdragon-yield-analytics (or similar)
- [ ] README contains all sections from section 11
- [ ] `requirements.txt` and `setup.sh` work on a clean Python 3.11+ environment
- [ ] `python data/generate_data.py && python data/setup_database.py && streamlit run ui/app.py` runs end-to-end without errors
- [ ] All three test scenarios produce coherent agent responses
- [ ] `pytest` runs with all tests passing
- [ ] A 2 to 3 minute demo video is recorded and linked in the README
- [ ] At least 3 screenshots are in `docs/demo_screenshots/`
- [ ] The architecture diagram is in `docs/architecture.png`
- [ ] The repo has been added to the portfolio website projects list
- [ ] Mention of this project is added to the resume under Projects
- [ ] The application to Qualcomm Job ID 3086988 has been submitted with the GitHub link included

---

## 15. Quick Reference: Key Numbers and Names

For consistency across the codebase, README, and any conversations about the project:

- **Project name**: AI-Powered Yield Analytics for Snapdragon SoC Production
- **Target product**: Snapdragon 8 Gen 5 (fictional, set in summer 2026)
- **Process node**: 3nm
- **NPU subsystem**: Hexagon NPU
- **Target NPU performance**: 50 TOPS
- **Pass threshold**: 48 TOPS, ≤3.5W power
- **Daily volume in demo**: 10,000 chips, 100 wafers
- **Normal yield**: ~95 percent
- **Problem-period yield**: ~68 percent
- **Problem trigger time**: 14:00 (hour 14)
- **Number of tools**: 5
- **Number of test scenarios**: 3
- **Demo runtime**: under 30 seconds per question
- **Build duration**: 7 days

---

End of brief.
