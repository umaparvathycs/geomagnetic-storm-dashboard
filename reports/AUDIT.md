# Audit of the uploaded project (all statements below were checked against the uploaded files)

Uploaded: `storm_catalogue.ipynb`, `Geomagnetic_plots.ipynb`, `generate_storm_plots1.py`, `storm_catalogue_1957-2026.csv`, `storm_plots_1957_2020.zip`.
**Not found in the uploaded files:** any Dst time-series CSV (`dst_master_unified_*`, `dst_dataset_1957-2026.csv`), any download/extraction script, the Belov et al. 2022 PDF, written teacher instructions (only the pasted project text exists), earlier documentation. Consequently the old outputs cannot be reproduced and the Dst values behind the old catalogue could not be inspected.

| Existing file | Purpose | Problems found | Decision |
|---|---|---|---|
| `storm_catalogue.ipynb` | Colab: load `dst_master_unified_*.csv`, detect storms, save catalogue | Data gap (below); trigger/backtrack/end rules (below); `errors="coerce"` + `dropna` silently drop rows so detection indexes across missing hours; default path `..._1957_present.csv` differs from uploaded `..._1957_2020.csv`; restricted to 1957-2020; output says 1521 storms | **Replace** (`src/storm_detection.py`) |
| `Geomagnetic_plots.ipynb` | Colab plotting from CSVs | Only the upload prompt is stored as output; parses timestamps with `format="mixed"` (different convention from notebook 1); no -150 content | **Replace** (`src/plotting.py`) |
| `generate_storm_plots1.py` | Batch plots for "all 2,355 storms" | Reads `dst_dataset_1957-2026.csv` (not uploaded); PNG named by `storm_id` so duplicate IDs overwrite each other; count hard-coded; `dayfirst=True` on ISO strings; boundaries come from the CSV, not a validated source | **Replace** |
| `storm_catalogue_1957-2026.csv` | 2355 storms 1957-01-02 .. 2026-09-09 | Defects below; no `-150` column; no data-quality flag | **Replace** (regenerate; keep as comparison reference only) |
| `storm_plots_1957_2020.zip` | 1521 PNGs | Matches the notebook's 1521-storm catalogue, not the 2355-row CSV | **Discard**, regenerate |

## Confirmed bugs / defects
1. **Duplicate storm IDs:** 27 surplus rows share an ID with another row (53 rows involved; 2328 unique IDs for 2355 rows), e.g. `GS_19670418_22`, `GS_19720224_07`. With plot filenames = ID, 27 plots are overwritten.
2. **Overlapping events:** 35 rows start at or before the previous row's end; 29 start before the previous row's *peak* (e.g. `GS_19700817_01` ends 1970-08-24 06:00 while `GS_19700822_07` starts 1970-08-22 07:00). Min end-to-start difference: -293 h.
3. **Missing years 2001-2004:** no storm starts in 2001-2004; the notebook output shows one gap of 35,065 h (2000-12-31 23:00 -> 2005-01-01 00:00). Kyoto's final-index page lists 2001-2004 months, so this is a data-acquisition failure, not an absence of storms.
4. **Mixed provenance:** notebook gives 1521 storms for 1957-2020, the zip has 1521 PNGs, but the CSV holds 2120 rows with start <= 2020 (573 of their IDs have no plot). The CSV therefore was **not** produced by the uploaded notebook code (a different, unseen version).
5. **Implausibly long events:** 145 storms last > 240 h; the March 1989 storm is a single event 1989-03-02 03:00 -> 1989-04-11 07:00 (964 h, min -589 nT) and the longest event is 1004 h. Cause in the notebook code: the end rule needs 24 consecutive hours >= -15 nT, so recurrent activity chains many storms into one.
6. **Onset back-tracking:** the notebook walks back from the trigger with `while start_idx > 0 and dst[start_idx] < -10` without a lower bound at the previous event's end or at data gaps -> a plausible mechanism for bug 2 (not proven, source Dst not available).
7. **Inconsistent timestamp parsing** across the three code files (`dayfirst=True, errors="coerce"` / `format="mixed"` / `dayfirst=True, format="mixed"`).
8. **`descent_rate` uses `max(main_phase, 1.0)`** - fabricates a 1 h denominator (7 catalogue rows have main phase <= 1 h).
9. Tier function in the notebook agrees with the CSV (0 mismatches). It is internally consistent and equals the boundary convention adopted here (labels differ: `Super-Intense` -> `Extreme`).

## Possible issues needing verification (cannot be decided without the Dst series)
- Whether the old catalogue mixes final/provisional/real-time data (Kyoto's final page covers 1957-2020 only; 2021-2026 must come from other products).
- Whether any old storm bridges a missing-data hour.
- Timestamp convention (hour start vs end) used when the old master CSV was built.
- Kyoto warns of missing data in 2017, 2018, 2020 (page "caution" link) - the old data were not checked against it.

## The -150 nT issue
Not found in the uploaded files: no code or CSV column calculates a recovery-to-(-150) value (the only "150" in code is `dpi=150`). The reported negative value cannot be traced; the new definition is implemented and tested (METHODOLOGY section 6).

## Missing components in the old project
Raw/clean data separation, data-quality report, validation suite, -150 metric, data-quality flags, CNN (any), chronological split, class-imbalance handling, event-level evaluation, living-catalogue update logic, tests, README, requirements, config.
