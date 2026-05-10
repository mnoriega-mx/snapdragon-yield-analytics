# Scenario validation report

Generated 2026-05-09 19:30 by `python scripts/validate_scenarios.py`.

Each scenario runs the live agent against the canonical question from the project brief and checks a handful of soft expectations (concept presence, tools called). Wording shifts run to run; the checks are designed to catch outright regressions, not enforce exact strings.

## Normal operation -- PASS (3/3)

**Question:** How is yield today?

Iterations: 2, duration: 11921 ms, tools called: query_database

- [x] calls query_database
- [x] does not start a deep investigation
- [x] answer mentions yield

**Final answer**

```
There's a clear and significant yield drop today (April 7). Here's the summary:

**Today's Yield: 84.5%** -- a major drop from the prior 6-day average of ~97.8%.

**What happened hour by hour:**
- Hours 00:00–13:00: Yield was normal, ranging from **96.4% to 98.6%**, consistent with the rest of the week.
- **14:00 onward: Yield collapsed**, dropping from 70.2% at 14:00 down to a low of **63.2% at 16:00**, and has stayed in the 64–68% range through end of day (23:00).

**Overall today:**
- 10,000 chips tested, **8,453 passed, 1,547 failed** -- versus roughly 225 failures on a typical day.
- The week-wide failure breakdown points strongly at **NPU throughput** as the dominant failure mode (2,627 of 2,881 total weekly failures are NPU TOPS below spec).

**Recommended next step:** This looks like a sustained excursion that started at 14:00. I'd suggest a deeper investigation into what changed at that hour -- I can run anomaly detection and SPC analysis on the afternoon window if you'd like.
```

## Anomaly investigation -- PASS (5/5)

**Question:** Why did yield drop today?

Iterations: 6, duration: 86212 ms, tools called: query_database, detect_anomalies, query_database, calculate_spc_metrics, generate_chart, generate_chart, write_summary_report

Charts generated: failure_timeline_20260509_192858_614729.png, corr_npu_tops_vs_npu_power_w_20260509_192917_095586.png

- [x] calls at least 3 tools
- [x] calls detect_anomalies
- [x] produces a structured report
- [x] references NPU
- [x] references the afternoon

**Final answer**

```
The investigation is complete and the report has been generated above.
```

**Report**

_Generated 2026-05-09 19:29 from Snapdragon production data._

## Findings

### Yield drop timing and magnitude

Today (April 7) ended with a daily yield of 84.5 percent -- the worst in the 7-day dataset, which averaged 97.8 percent on the preceding six days. The line ran normally through the first 13 hours; per-hour yield held between 96.4 and 98.6 percent from midnight through 13:00. Then, starting at 14:00, yield collapsed to 70.2 percent and continued to deteriorate, reaching a floor of 64.2 percent at 23:00. All ten hours from 14:00 through end-of-day were flagged as anomalous, with per-hour failure rates ranging from 29.8 percent (14:00) to 36.7 percent (16:00). In absolute terms, 1,391 chips failed in that 10-hour window out of 4,163 tested -- a failure rate of 33.4 percent compared to a baseline of roughly 2.7 percent in the morning.

### Dominant failure mode

Of the 1,391 chips that failed in the afternoon window, almost all were flagged for NPU throughput falling below the 48 TOPS pass threshold: 1,347 chips, or 96.8 percent of afternoon failures, carried this failure reason. The remaining 44 were split across NPU power draw above spec (33 chips) and a handful of memory bandwidth and CPU frequency failures. This near-total concentration on a single failure mode strongly points to a single change in the NPU subsystem rather than a broad process excursion.

{{chart:failure_timeline}}

### NPU throughput and power correlation

The SPC analysis confirms the shift quantitatively. The hourly mean NPU throughput was rock-stable at approximately 50.5 TOPS from midnight through 13:00, then dropped abruptly to roughly 47.4--48.0 TOPS beginning at 14:00 and remained there for every subsequent hour. The hourly NPU throughput mean tracked the failure rate with a Pearson correlation of -0.9986 (essentially perfect inverse), and hourly NPU power tracked it in the opposite direction with r = +0.9983. This means that as chips lost throughput they simultaneously drew more power, which is the classic signature of an NPU that is working harder (or less efficiently) to process the same workload. CPU frequency (r = -0.19), memory bandwidth (r = +0.24), and die temperature (r = +0.13) were all near zero, exonerating those subsystems.

{{chart:correlation_chart:npu_tops:npu_power_w}}

### Onset and persistence

The excursion onset is abrupt and persistent rather than gradual. The NPU throughput mean went from 50.6 TOPS at 13:00 to 48.0 TOPS at 14:00 in a single step, and never recovered through the end of the 24-hour period (23 hours showed 47.5 TOPS). This rules out a transient upset and instead points to a sustained change that was present for the last 10 hours tested. The shift also spans every wafer processed after 14:00 -- starting with W658 at 14:00 and continuing sequentially through W699 at 23:59, covering every lot in the afternoon run.

## Root cause hypothesis

The evidence converges on a sustained NPU-subsystem excursion that began at or just before 14:00 on April 7. The near-perfect anti-correlation between hourly NPU throughput and failure rate (r = -0.9986), combined with a simultaneous rise in NPU power draw (r = +0.9983), indicates that the Hexagon NPU is delivering less computational output while consuming more power. Because the shift is abrupt, persistent, affects all wafers in the afternoon run, and leaves CPU, memory, and thermal metrics unaffected, the most likely explanation is a process or tooling change that degraded NPU execution efficiency -- such as a shift in the test or conditioning environment, a change in the incoming wafer population, or a step in the fab flow applied exclusively to lots entering the line around 14:00. The root cause is localized to the NPU compute path.

## Recommendations

1. Immediately quarantine all wafers processed after 14:00 (W658--W699, approximately 4,200 chips) pending further investigation, and halt additional wafers entering this production window until the root cause is identified.
2. Audit the production log for any process, tooling, recipe, or material change introduced between 13:00 and 14:00 on April 7 -- this is the most likely time window for the triggering event.
3. Run SPC on NPU power draw broken down by wafer ID for the afternoon lots to see whether the elevated power signature is uniform across all wafers or concentrated in a specific subset, which would narrow the scope of the excursion.
4. Re-test a representative sample of the quarantined chips under controlled conditions to distinguish test-environment artifacts from actual device degradation before making a yield disposition decision.
5. Compare the detailed NPU throughput distribution (not just the mean) of morning versus afternoon chips to quantify whether all chips shifted equally or whether there is a bimodal population, as this would help identify whether the excursion is wafer-level or lot-level.

## Specific lookup -- PASS (2/2)

**Question:** Show me Wafer W050's performance.

Iterations: 2, duration: 13789 ms, tools called: query_database

- [x] calls query_database
- [x] answer references W050

**Final answer**

```
Here's a full performance summary for Wafer W050:

---

## Wafer W050 -- Performance Summary

**Tested:** 2026-04-01 | **Process Node:** 3nm | **SoC Model:** SD8Gen5
**Total Chips:** 100 | **Passes:** 97 | **Fails:** 3 | **Yield: 97.0%**

---

### Failures (3 chips)

All 3 failures were due to **NPU throughput below spec** (< 48 TOPS):

| Chip ID | NPU TOPS | CPU Freq (GHz) | NPU Power (W) | Mem BW (GB/s) | Die Temp (°C) |
|---------|----------|---------------|---------------|----------------|----------------|
| C05025 | 47.35 | 3.387 | 3.193 | 206.2 | 74.5 |
| C05044 | 47.95 | 3.371 | 3.240 | 199.4 | 78.6 |
| C05066 | 47.96 | 3.391 | 3.305 | 202.1 | 76.0 |

All other metrics on the failing chips were within spec. The failures are scattered across the wafer (chips 25, 44, and 66), suggesting isolated NPU throughput excursions rather than a systematic spatial defect.

---

### Passing Chip Metrics (97 chips)

| Metric | Min | Max | Approx. Mean |
|--------|-----|-----|--------------|
| NPU TOPS | 48.24 | 53.27 | ~50.5 |
| NPU Power (W) | 2.98 | 3.49 | ~3.21 |
| CPU Freq (GHz) | 3.34 | 3.55 | ~3.45 |
| Mem BW (GB/s) | 193.0 | 215.4 | ~205.0 |
| Die Temp (°C) | 69.5 | 85.1 | ~78.7 |

---

### Summary

W050 is a **healthy wafer** at 97.0% yield. The only anomaly is 3 chips with marginally low NPU throughput (47.35--47.96 TOPS, just below the 48 TOPS threshold). All other subsystems -- CPU, memory bandwidth, NPU power, and die temperature -- are comfortably within spec across the board. No action is required unless this NPU throughput pattern repeats across neighboring wafers.
```
