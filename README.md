# Geomagnetic Storm Identification & Living Storm Catalogue (Kyoto WDC Dst)

Automated geomagnetic storm detection using a 1D Convolutional Neural Network
trained on 64 years of hourly Kyoto Dst observations (1957–2020). Includes a
deterministic reference catalogue, CNN-based classification, event-level
evaluation, and a monthly update pipeline for a living catalogue.

## Project Status

| Component | Status |
|---|---|
| Kyoto WDC Dst download & fixed-width parser | **Implemented & verified** against real Kyoto files |
| Data cleaning & hourly grid validation | **Verified**: 561,024 final hours, zero gaps, zero NaNs |
| Deterministic storm detection & catalogue | **Verified**: 4,824 storms (1957–2020) with recovery-to-150 |
| 1D-CNN training, evaluation, saved checkpoint | **Trained & frozen** (`outputs/models/storm_cnn.pt`, best epoch 6) |
| Hourly test set metrics (2016–2020, 43,801 windows) | **Accuracy 97.25%, F1 89.77%, PR-AUC 95.81%** |
| Event-level evaluation (1-to-1 Hungarian matching) | **Precision 63.51%, Recall 70.70%, F1 66.91%** |
| Temporal overlap coverage diagnostic | **100% of 256 test reference storms overlap CNN predictions** |
| Automated test suite | **40 tests, all passing** |
| Monthly catalogue update pipeline | **Implemented** (`src/update_catalogue.py`) |
| Early-warning intensity prototype (6-hour onset) | **Prototype** (88.72% accuracy; see limitations below) |

## 🚀 Live Dashboard

Access the deployed application here:

🔗 **[Geomagnetic Storm Identification – Live Dashboard](https://geomagnetic-storm-dashboard-8pvf3yyrh4a6vnqvrdhn37.streamlit.app/)**

## Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Run the complete pipeline (one command)

```bash
python run_project.py
```

This executes four stages sequentially:
1. Data integrity verification (561,024 final Dst hours)
2. Full test suite (40 automated tests)
3. CNN hourly evaluation on the 2016–2020 test set
4. Event-level evaluation with Hungarian matching and coverage diagnostics

### Run individual components

```bash
python verify_preprocessing.py              # Pre-flight data check
python -m unittest discover tests -p "test_*.py"  # All 40 tests
python -m src.evaluate_cnn                  # Hourly CNN metrics
python -m src.evaluate_events               # Event-level metrics & plots
```

### Launch the interactive dashboard (optional)

```bash
streamlit run dashboard_app.py
```

## Data

- **Source:** [Kyoto WDC Dst Index](https://wdc.kugi.kyoto-u.ac.jp/dst_final/index.html)
- **Period:** 1957-01 through 2020-12 (final product only)
- **Cadence:** Hourly (561,024 observations)
- **Storm trigger:** Dst ≤ −30 nT
- **Storm release:** Dst ≥ −20 nT
- **Merge gap:** 6 hours

## Model

- **Architecture:** 1D-CNN (Conv1d 1→16→32→64, BatchNorm, ReLU, AdaptivePool)
- **Input:** 48-hour trailing window of Dst, scaled by 100 nT
- **Training period:** 1957–2009
- **Validation period:** 2010–2015
- **Test period:** 2016–2020 (43,801 hourly windows)
- **Checkpoint:** `outputs/models/storm_cnn.pt` (best epoch 6, early stopping at 11)

## Verified Results

### Hourly classification (threshold 0.5, test set 2016–2020)

| Metric | Value |
|---|---|
| Accuracy | 97.25% |
| Precision | 89.73% |
| Recall | 89.81% |
| F1 | 89.77% |
| PR-AUC | 95.81% |
| Confusion | TN=37,320  FP=604  FN=599  TP=5,278 |

### Event-level evaluation (strict 1-to-1 Hungarian matching)

| Metric | Value |
|---|---|
| Reference storms | 256 |
| Predicted events | 285 |
| Event precision | 63.51% |
| Event recall | 70.70% |
| Event F1 | 66.91% |

### Supplementary temporal coverage diagnostic

256 of 256 reference storms (100%) have temporal overlap with at least one CNN
predicted storm interval. This is reported as a **coverage diagnostic**, not as
event recall. The gap between 70.70% strict recall and 100% overlap coverage
is explained by 50 long CNN predictions that span multiple closely-spaced
reference sub-storms. The Hungarian algorithm assigns each predicted event to
at most one reference event, classifying the remaining overlapping storms as
unmatched.

## Early-Warning Prototype (Experimental)

`src/early_warning_model.py` trains a Gradient Boosting classifier to predict
the final peak intensity tier (Weak / Moderate / Intense+) from the first 6
hours of a storm's onset trajectory. This is a **prototype feasibility study**:

- Overall accuracy: 88.72% (dominated by Weak storms: 197 of 257 test storms)
- Intense+ class: only 5 test storms (precision 40%, recall 40%)
- Lead-time claim requires further validation (see limitations)

## Scientific Limitations

1. **Dst-to-Dst self-consistency:** The reference catalogue is derived from Dst
   and the CNN is trained on Dst-derived labels. This project evaluates how
   well the CNN reproduces the specified Dst-based storm definition. It does
   not independently prove physical storm detection using an independent
   measurement source.

2. **Retrospective classification, not forecasting:** The CNN uses a 48-hour
   trailing window of *observed* Dst. It is a real-time pattern recognizer, not
   an advance forecasting system.

3. **Early-warning prototype limitations:** The 6-hour intensity predictor is
   tested on only 5 Intense+ storms (2016–2020). Per-class performance on rare
   intense storms is not statistically robust.

## Layout

```
config.yaml           All parameters
src/                   Source modules
tests/                 40 automated tests
data/raw/              Downloaded Kyoto HTML files
data/processed/        Cleaned hourly CSV (dst_hourly_clean.csv)
data/catalogue/        Storm catalogue (storm_catalogue.csv)
outputs/models/        CNN checkpoint (storm_cnn.pt)
outputs/metrics/       Evaluation CSVs and JSON
outputs/figures/       Annual validation plots
reports/               Audit and research reports
run_project.py         One-click master pipeline runner
dashboard_app.py       Interactive Streamlit dashboard (optional)
```

## Reproducibility

Fixed seed (`cnn.seed: 42`), deterministic flags, all parameters in
`config.yaml`, chronological splits (no random shuffle), fixed-constant
scaling. Record `pip freeze` with your results.
