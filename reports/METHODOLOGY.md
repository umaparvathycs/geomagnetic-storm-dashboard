# Methodology (report-ready). Claims about real data are marked "TO FILL" until you run the pipeline.

## 1. Data
Source: Kyoto WDC Dst index, hourly, UT. The FINAL product page lists 1957-01 to 2020-12 only (checked on the page); later months come from the provisional and real-time products, which are less reliable and may be revised. The pipeline records the product of every hour and gives precedence final > provisional > real-time. Kyoto's page warns of missing data in 2017, 2018 and 2020.
TO FILL: exact period, hours expected/valid/missing, list of gaps > 24 h, from `data_quality_report.json`.

## 2. Raw vs clean; interpolation
RAW (`dst_raw_hourly.csv`) keeps every parsed value, including the 9999 missing marker. CLEAN builds a complete hourly grid; 9999 and |Dst| > 1000 nT become NaN; duplicates are resolved by product precedence and counted. **No interpolation by default** (`max_interp_gap_hours: 0`): an interpolated hour could create or hide a threshold crossing or a minimum. If enabled, only interior gaps up to that length are filled and each filled hour is flagged `interpolated`. Gaps are never bridged by detection.
Assumptions to verify on real files (config switches): hourly value k of a day is stamped `day 00:00 + (k-1) h` (`timestamp_convention`), and the base-value column is not added (`add_base_value: false`). Verification test: March 1989 minimum must equal -589 nT (value from the uploaded catalogue).

## 3. Storm definition (parameters in `config.yaml`)
| Element | Rule | Reason |
|---|---|---|
| Trigger | Dst <= -30 nT (teacher) | Given |
| Event | maximal sub-threshold run; runs separated by <= 6 non-trigger hours with no missing data are merged | Hourly noise around -30 must not split one storm; multiple minima stay one event, peak = global minimum (earliest on ties). 6 h is a parameter; see sensitivity table |
| Min size | >= 1 trigger hour (literal teacher rule) | Single-hour events are kept and can be filtered with `min_core_hours` |
| Onset | walk back from the first trigger hour while Dst is strictly higher one hour earlier; max 24 h; never beyond the previous event's end or a data gap | Onset = start of the monotone main-phase descent, bounded so events cannot overlap |
| End | first hour after the core with Dst >= -20 nT | 10 nT hysteresis; avoids the old 24 h/-15 nT chaining |
| Closely spaced storms | if Dst never reaches -20 nT between two events, the boundary is the hour of maximum Dst between them (end of the first; next starts the following hour) | Splits without overlap, no value assigned twice |
| Missing data / series end | event truncated and flagged `end_truncated_by_gap`, `onset_truncated_by_gap`, `open_end`, `start_at_data_edge` | Nothing invented; `open_end` storms are provisional in the living catalogue |
| IDs | `GS_YYYYMMDD_HH` of the start hour, created after boundaries are final | Non-overlap implies uniqueness |

Intervals are inclusive of both end hours. `total = end-start`, `main = peak-start`, `recovery = end-peak` (hours); `descent_rate = (onset Dst - min Dst)/main` (NaN if main = 0, no artificial denominator).

## 4. Intensity tiers
The teacher's ranges overlap (Intense to -250 vs Severe from -200/-250; Severe to -350 vs Extreme below -250/-350). Assumption used everywhere (confirm with the teacher): each tier is (lower, upper] on the minimum Dst:
Weak (-50,-30], Moderate (-100,-50], Intense (-250,-100], Severe (-350,-250], Extreme <= -350. So -50 is Moderate, -100 Intense, -250 Severe, -350 Extreme. Changing it needs only `config.yaml` (and a re-run).

## 5. Validation (automatic, fails loudly)
Unique IDs, no duplicate/overlapping intervals, chronological order, start <= peak <= end (equalities only with a justifying flag), non-negative and consistent durations, minimum equals the series value at the peak and the interval minimum, peak = first minimum, no missing hour inside an event, tier consistent with minimum, -150 metric consistent (below), recomputation of -150 from the series.

## 6. Recovery to -150 nT
`recovery_to_150_hours` = time from the **peak** to the first **upward** crossing of -150 nT **after the peak** (linear interpolation between the bracketing hours). Only storms with min <= -150 nT get a value; others, and storms truncated before recovery, get NaN. The pre-peak downward crossing is never used. It differs from `recovery_phase_hours` (peak -> -20 nT end). The validator enforces 0 < value <= recovery_phase.

## 7. CNN task
Binary window classification, W = 48 h, stride 1. Label: 1 iff the window's last hour lies in a catalogue storm interval (causal; `label_position: center` is an offline option). Windows containing a missing hour are skipped. Rationale for 48 h: covers a typical main phase plus the first part of recovery and the preceding quiet baseline; TO FILL: compare with 24/72 h if time permits (not tested).
Architecture: Conv1d(16,k5)-BN-ReLU-Pool, Conv1d(32,k5)-BN-ReLU-Pool, Conv1d(64,k3)-BN-ReLU, global mean+max pooling, Dense(64)-Dropout(0.3)-logit. Small on purpose: one input channel, short windows.
Split (chronological, `config.yaml`): train 1957-2009, validation 2010-2015, test 2016-end. A window is assigned only if all 48 hours lie in one split, so no hour is shared. Random splitting of stride-1 windows would put near-identical windows in train and test (leakage) and would also mix solar-cycle regimes. Scaling by a fixed constant (Dst/100).
Imbalance: report prevalence per split; use `pos_weight = neg/pos` from the training split in the loss; no duplication of data. TO FILL: class distribution table (`class_distribution.csv`).
Decision threshold: chosen on validation (max F1) and applied unchanged to test.

## 8. Evaluation
Window level: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix. Event level (test period): fraction of catalogue storms overlapped by a predicted segment (any overlap and >= 50 % coverage), onset delay, predicted segments with no catalogue storm (false alarms). **Baseline reported alongside:** "storm iff last hour <= -30 nT". Because the catalogue itself uses -30 nT, a good CNN score is not evidence of learning beyond that rule; the comparison with the baseline is the honest test. The CNN does not validate the catalogue: the catalogue produces the labels, the CNN is a learned recogniser compared against them.
TO FILL: all numbers; do not report accuracy alone; describe failures (likely: onset hours, recovery tail, weak storms) from `event_recall_by_tier.csv` and the example figures.

## 9. Living catalogue
`update_catalogue`: download new months -> rebuild raw/clean -> data validation -> deterministic re-detection -> catalogue validation -> diff (new/changed/removed/unchanged) -> protected closed final-data storms may not change without `--accept-changes` -> previous catalogue archived in `data/catalogue/history/` -> only new/changed plots redrawn (hash manifest) -> optional CNN inference on the last ~2 months -> append to `update_log.jsonl`. Storms with `open_end` or from provisional/real-time data are expected to change and are replaced.

## 10. Limitations
Detection depends on rule parameters (run `--sensitivity`); Dst final vs provisional differ; hour-start convention unverified; CNN labels inherit catalogue choices; storms at the series edge are provisional; test period has few extreme storms so tier-level CNN statistics will be noisy; CNN code unexecuted until you run it.
## 11. Future work
Multi-class tiers; per-hour sequence labelling (1D U-Net/TCN) instead of window classification; storm-phase segmentation; adding solar-wind/IMF inputs; threshold-free event scoring; comparison with the Belov et al. approach once the paper is supplied (not available here).
## 12. Demo outline (5 min)
1 problem + data (Kyoto) -> 2 old problems found (audit table) -> 3 rule diagram on one real storm plot -> 4 catalogue stats -> 5 chronological split figure + CNN curves -> 6 CNN vs baseline vs catalogue (event level) -> 7 live monthly update run.
