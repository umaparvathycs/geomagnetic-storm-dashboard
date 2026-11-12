# Scientific Research Report: 1D-CNN Geomagnetic Storm Recognition and Event Evaluation

**Project Title:** Deterministic Geomagnetic Storm Catalogue Construction and 1D-CNN Sequence Recognition from Kyoto Dst Index  
**Authors/Auditors:** Machine Learning and Space Physics Pair Programming Team  
**Date:** October 28, 2026  
**Repository:** `geomag_storm_project`  

---

## 1. Research Objective & Scientific Context

Geomagnetic storms are major disturbances of the Earth's magnetosphere caused by enhanced solar wind energy and magnetic coupling. The disturbance storm time (Dst) index—monitored and distributed by the World Data Center (WDC) for Geomagnetism, Kyoto—measures equatorial geomagnetic depression caused by the intensification of the magnetospheric ring current.

The objectives of this project are:
1. Deterministically preprocess Kyoto WDC hourly Dst observations and extract a verified historical storm catalogue for the final observation period (1957–2020).
2. Train a 1D Convolutional Neural Network (CNN) to recognize storm-related hourly states from 48-hour trailing Dst time-series profiles.
3. Rigorously evaluate hourly classification and event-level storm detection performance under leak-free chronological splitting.
4. Establish scientifically defensible event-level matching and post-processing frameworks that reconcile hourly predictions with physical storm episodes.

**Fundamental Scientific Caveat:**  
Because the reference storm catalogue is constructed deterministically from Kyoto Dst and the CNN is trained using catalogue-derived window labels, this project evaluates the capacity of a 1D-CNN to reproduce the mathematical and morphological criteria of the Dst storm definition. It does not constitute independent physical discovery from exogenous measurement sources (such as interplanetary magnetic field $B_z$, solar wind velocity, or ground-level cosmic ray detectors).

---

## 2. Dataset and Preprocessing Methodology

### 2.1 Kyoto WDC Observations
- **Data Product:** Final Kyoto Dst (`dst_final`) only.
- **Temporal Coverage:** 1957-01-01 00:00:00 UTC through 2020-12-31 23:00:00 UTC (64 complete calendar years).
- **Total Valid Observations:** Exactly $561,024$ hourly rows.
- **Completeness:** Zero duplicate timestamps, zero missing hours, zero NaNs in the final study series.
- **Excluded Data:** Provisional data (`dst_provisional`, 48,912 hours from 2021 to 2026) were parsed and stored in processed raw archives but strictly excluded from model training, validation, and testing.

### 2.2 Kyoto Fixed-Width and HTML Parsing
Kyoto records are formatted with negative integer concatenation (e.g. `14 -583-589-463...` for the March 1989 storm). The parser utilizes robust regex signed integer tokenization and fixed-width extraction (`DSTyyMM*ddRRX0ccBASE...`), ensuring verified retrieval of extreme events (e.g. March 14, 1989 minimum of $-589\text{ nT}$ at 01:00 UT).

---

## 3. Deterministic Storm Definition & Catalogue

The reference catalogue is generated according to the following deterministic rules:
1. **Core Trigger:** Consecutive hours where $\text{Dst} \le -30.0\text{ nT}$.
2. **Core Merging:** Runs separated by $\le 6$ non-trigger hours with no missing observations are merged into a single core.
3. **Minimum Core Duration:** $\ge 1$ hour.
4. **Onset Search:** Walking backward from the first core hour while $\text{Dst}[i-1] > \text{Dst}[i]$, bounded by 24 hours lookback, data gaps, or the preceding event boundary.
5. **Event Termination:** The first hour following the core where $\text{Dst} \ge -20.0\text{ nT}$ (release threshold). If a subsequent core begins before reaching $-20\text{ nT}$, the event is closed at the maximum Dst between the two cores (`split_without_recovery`).
6. **Recovery to -150 nT:** Measured from the storm minimum (peak) to the first *upward* crossing of $-150\text{ nT}$ strictly after the peak, utilizing fractional linear interpolation.

**Reference Catalogue Statistics (1957–2020):**
- Total Storms: $4,824$
- Weak ($-50 < \text{Dst}_{\min} \le -30\text{ nT}$): $3,292$ ($68.2\%$)
- Moderate ($-100 < \text{Dst}_{\min} \le -50\text{ nT}$): $1,165$ ($24.2\%$)
- Intense ($-250 < \text{Dst}_{\min} \le -100\text{ nT}$): $330$ ($6.8\%$)
- Severe ($-350 < \text{Dst}_{\min} \le -250\text{ nT}$): $27$ ($0.56\%$)
- Extreme ($\text{Dst}_{\min} \le -350\text{ nT}$): $10$ ($0.21\%$)
- Storms reaching $\le -150\text{ nT}$ with valid recovery times: $135$ (range: $0.00$ to $23.86$ hours, median: $4.79$ hours). Zero negative values.

---

## 4. CNN Architecture & Training Setup

### 4.1 Model Structure
The neural network (`StormCNN`) is a 1D temporal convolutional architecture operating on trailing 48-hour windows:
- Input: `(batch, 48, 1)` representing $[\text{Dst}_{t-47}, \dots, \text{Dst}_t]$ divided by fixed scale $100.0\text{ nT}$.
- Feature Extractor:
  - `Conv1d(1, 16, k=5, pad=2)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` (24 steps)
  - `Conv1d(16, 32, k=5, pad=2)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` (12 steps)
  - `Conv1d(32, 64, k=3, pad=1)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` (12 steps)
- Aggregation: Dual Global Pooling (concatenation of Global Max Pooling and Global Average Pooling $\rightarrow$ 128 dimensions).
- Classification Head: `Linear(128, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(64, 1)`.

### 4.2 Chronological Split & Leakage Controls
- **Training Set (1957–2009):** 464,545 windows (134,801 positive, 329,744 negative; prevalence $29.02\%$).
- **Validation Set (2010–2015):** 52,537 windows (11,017 positive, 41,520 negative; prevalence $20.97\%$).
- **Test Set (2016–2020):** 43,801 windows (5,877 positive, 37,924 negative; prevalence $13.42\%$).
- **Isolation Constraint:** Every hour of a candidate 48-hour window must fall strictly within the split boundaries. Test window evaluation begins on 2016-01-02 23:00 UTC, ensuring zero overlap with the validation period.

---

## 5. Verified Performance Results

### 5.1 Hourly Classification Performance (Held-Out Test Set: 2016–2020)

| Metric | Verified Value | Benchmark / Baseline |
|---|---:|---:|
| Test Windows Evaluated | 43,801 | 43,801 |
| Classification Accuracy | 97.25% | 97.25% |
| Precision | 89.73% | 89.73% |
| Recall | 89.81% | 89.81% |
| F1 Score | 89.77% | 89.77% |
| PR-AUC (Average Precision) | 95.81% | 95.81% |
| True Negatives (TN) | 37,320 | 37,320 |
| False Positives (FP) | 604 | 604 |
| False Negatives (FN) | 599 | 599 |
| True Positives (TP) | 5,278 | 5,278 |

The model achieves balanced, near-90% precision and recall on the imbalanced held-out test distribution without retraining.

---

### 5.2 Event-Level Detection Performance

Hourly model predictions ($p \ge 0.5$) are converted into discrete predicted events by merging positive runs with gaps $\le 6$ hours.

#### A. Strict One-to-One Hungarian Bipartite Matching (Primary Isolated Storm Metric)

| Metric | Verified Value | Scientific Interpretation |
|---|---:|---|
| Evaluation Coverage | 2016-01-02 23:00 to 2020-12-31 23:00 UTC | Valid 48-hour test window span |
| Reference Storms | 256 | Distinct catalogue storm intervals |
| CNN Predicted Events | 285 | Merged positive prediction segments |
| Detected Storms (TP) | 181 | 1-to-1 matched pairs ($\ge 1\text{ h}$ overlap) |
| Missed Storms (FN) | 75 | Unassigned reference storms under 1-to-1 |
| False Detections (FP) | 104 | Unassigned predicted events under 1-to-1 |
| Event Precision | 63.51% | $\text{TP} / \text{Predicted}$ |
| Event Recall | 70.70% | $\text{TP} / \text{Reference}$ |
| Event F1 Score | 66.91% | Harmonic mean of precision and recall |

#### B. Physical Storm Recognition & Coverage (Episode / Multi-Dip Metric)

| Metric | Verified Value | Scientific Interpretation |
|---|---:|---|
| Reference Storms with Overlap | **256 / 256 (100.0%)** | All 256 storms overlapped positive predictions |
| Missed Storms with Zero Overlap | **0 / 256 (0.0%)** | Zero reference storms were truly missed |
| Storms with $\ge 50\%$ Duration Coverage | 252 / 256 (98.4%) | High-fidelity temporal coverage |
| Storms with $\ge 80\%$ Duration Coverage | 203 / 256 (79.3%) | Substantial morphological coverage |
| Complete Duration Coverage (100%) | 109 / 256 (42.6%) | Entire catalogue interval predicted |
| Predictions Spanning Multi-Storms | 50 predicted events | Compound disturbance episodes spanning 125 sub-storms |

#### C. Timing and Duration Fidelity (Matched Events)
- **Onset Delay:** Mean $-1.26\text{ h}$, Median $+2.0\text{ h}$, Std $14.23\text{ h}$ (model onset closely tracks main-phase start).
- **Duration Error:** Mean $+7.37\text{ h}$, Median $0.0\text{ h}$, Std $20.59\text{ h}$ (slight tendency to extend into quiet recovery).
- **Temporal Overlap:** Mean $23.71\text{ h}$, Median $14.0\text{ h}$, Min $1\text{ h}$, Max $177\text{ h}$.

---

### 5.3 Performance Disaggregation

#### By Intensity Tier:

| Intensity Tier | Reference Count | 1-to-1 Matched | 1-to-1 Recall | Recognition Coverage |
|---|---:|---:|---:|---:|
| Weak ($-50 < \text{Dst}_{\min} \le -30\text{ nT}$) | 196 | 129 | 65.82% | 100.0% |
| Moderate ($-100 < \text{Dst}_{\min} \le -50\text{ nT}$) | 55 | 48 | 87.27% | 100.0% |
| Intense ($-250 < \text{Dst}_{\min} \le -100\text{ nT}$) | 5 | 4 | 80.00% | 100.0% |
| Severe / Extreme | 0 | 0 | N/A (Solar Minimum) | N/A |

*Insight:* Stronger storms achieve substantially higher 1-to-1 recall ($87.3\%$ for Moderate, $80.0\%$ for Intense). Weak storms near the $-30\text{ nT}$ boundary account for 67 out of the 75 missed 1-to-1 storms.

#### By Calendar Year (2016–2020):

| Year | Reference Storms | Predicted Events | 1-to-1 Matched | 1-to-1 Precision | 1-to-1 Recall | Coverage Recall |
|---|---:|---:|---:|---:|---:|---:|
| 2016 | 67 | 71 | 45 | 63.38% | 67.16% | 100.0% |
| 2017 | 77 | 82 | 59 | 71.95% | 76.62% | 100.0% |
| 2018 | 40 | 50 | 26 | 52.00% | 65.00% | 100.0% |
| 2019 | 34 | 44 | 24 | 54.55% | 70.59% | 100.0% |
| 2020 | 38 | 38 | 27 | 71.05% | 71.05% | 100.0% |

#### Physical Analysis of the 104 "False Detections":
- $\text{Dst}_{\min} \le -30.0\text{ nT}$: Exactly 0 ($0\%$).
- Sub-threshold depression ($-29 \le \text{Dst}_{\min} \le -25\text{ nT}$): 76 events ($73.1\%$).
- Weak/quiet ($-25 < \text{Dst}_{\min} \le +3\text{ nT}$): 28 events ($26.9\%$).
- Median Duration: 1.0 hour.  
*Scientific Conclusion:* The majority of false alarms are genuine magnetospheric depressions that narrowly missed the arbitrary $-30\text{ nT}$ catalogue threshold.

---

### 5.4 Merge-Gap Sensitivity Analysis

| Merge Gap | Predicted Events | 1-to-1 Precision | 1-to-1 Recall | 1-to-1 F1 | Multi-Ref Preds | Ref Coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 h | 457 | 49.45% | 88.28% | 63.39% | 28 | 100.0% |
| 1 h | 388 | 56.96% | 86.33% | 68.63% | 31 | 100.0% |
| 2 h | 356 | 60.39% | 83.98% | **70.26%** | 33 | 100.0% |
| 3 h | 333 | 61.26% | 79.69% | 69.27% | 40 | 100.0% |
| 4 h | 310 | 61.29% | 74.22% | 67.14% | 47 | 100.0% |
| 5 h | 293 | 62.12% | 71.09% | 66.30% | 50 | 100.0% |
| **6 h (Default)** | **285** | **63.51%** | **70.70%** | **66.91%** | **50** | **100.0%** |
| 8 h | 261 | 64.75% | 66.02% | 65.38% | 54 | 100.0% |
| 12 h | 223 | 66.82% | 58.20% | 62.21% | 57 | 100.0% |

*Conclusion:* Gap = 2h yields the optimal balance for isolated 1-to-1 matching ($F_1 = 70.26\%$). However, preserving the configured 6-hour gap aligns directly with catalogue sub-core merging definitions.

---

## 6. Literature Context: Belov et al. (2022)

In cosmic-ray and space weather literature, Belov et al. (2022) investigated geomagnetic storm recognition using ground-based cosmic-ray observations from neutron monitors and muon telescopes. They reported an event-recognition probability of $\sim 82.3\%$ at a false-alarm rate of $10\%$, and $\sim 71.7\%$ at a false-alarm rate of $\le 5\%$.

**Critical Methodological Distinction:**  
The Belov et al. (2022) study evaluates *independent physical prediction from galactic cosmic-ray anisotropy and modulation*, anticipating Dst disturbances prior to ground-level geomagnetic depression. In contrast, the current project trains a 1D-CNN on *historical Dst observations to classify Dst-derived storm intervals*. Consequently, quantitative comparisons between the two are scientifically incomparable and should not be presented as benchmark superiority.

---

## 7. Conclusions & Recommendations

1. **Check-and-Balance Verification:** The project codebase is structurally sound, mathematically verified, and fully reproducible. No retraining of the CNN is required.
2. **Evaluation Dual-Reporting:** Time-series event detection must report both strict 1-to-1 isolated storm metrics ($\text{Recall} = 70.70\%$, $\text{Precision} = 63.51\%$, $F_1 = 66.91\%$) and physical temporal overlap coverage ($100.0\%$ overlap rate, $98.4\%$ with $\ge 50\%$ duration coverage) as a separate diagnostic. This truthfully conveys performance on compound space weather episodes without confusing overlap coverage with bijective event recall.
3. **Future Extension:** Replacing window-level classification with sequence-to-sequence segmentation (e.g. 1D U-Net or Temporal Convolutional Networks) and incorporating upstream ACE/DSCOVR solar wind magnetic field inputs ($B_z$) will elevate the system from retrospective Dst pattern recognition to true early-warning forecasting.
