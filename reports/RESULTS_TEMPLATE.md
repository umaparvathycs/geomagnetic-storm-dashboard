# Results (fill from the generated files; every number must come from a file in outputs/ or data/)

## 1 Data (data/processed/data_quality_report.json)
Period ____ to ____; expected hours ____; valid ____; missing ____; gaps > 24 h: ____. Hours by product: final ____ / provisional ____ / real-time ____. Parser check (March 1989 = -589 nT): PASS / FAIL.

## 2 Catalogue (data/catalogue/storm_catalogue.csv)
Storms: ____. By tier: Weak ____, Moderate ____, Intense ____, Severe ____, Extreme ____. Validation: all checks pass (`python -m src.storm_detection` completed). Flags: ____ storms with non-"ok" flag. Storms reaching -150 nT: ____; with post-peak crossing value: ____.
Sensitivity (outputs/metrics/detection_sensitivity.csv): count ranges from ____ to ____ across the parameter grid.
Case studies (>= 4, incl. weak and strong): date, catalogue min vs Kyoto page, boundaries plausible? ____

## 3 CNN (outputs/metrics/*, outputs/figures/*)
Class distribution per split (class_distribution.csv): ____. Training curves: training_curves.png.
Test, window level (metrics.json): accuracy ____ precision ____ recall ____ F1 ____ ROC-AUC ____ PR-AUC ____; baseline rule: ____.
Test, event level: storms scored ____; recall (any overlap) ____; recall (>=50 % coverage) ____; false-alarm segments ____; median onset delay ____ h; by tier (event_recall_by_tier.csv).
Honest interpretation: does the CNN beat the baseline? Where does it fail (onset, recovery tail, weak storms)? ____

## 4 Final quality checklist (tick only after verifying)
- [ ] Storm IDs unique  - [ ] no overlaps  - [ ] start <= peak <= end  - [ ] durations consistent
- [ ] min Dst = value at peak  - [ ] tiers consistent  - [ ] -150 crossing is post-peak, NaN when unreached
- [ ] plot count = catalogue rows  - [ ] CNN labels from catalogue  - [ ] no split overlap (tests pass)
- [ ] test = latest unseen years  - [ ] model reloads and `--infer-since` works  - [ ] monthly update ran once on real data
