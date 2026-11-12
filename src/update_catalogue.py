"""LIVING CATALOGUE UPDATE (run monthly):
   python -m src.update_catalogue                    # download new months, rebuild, validate, diff, plot
   python -m src.update_catalogue --skip-download    # only reprocess files already in data/raw
   python -m src.update_catalogue --csv new.csv --product dst_provisional   # merge a CSV instead of HTML
   python -m src.update_catalogue --accept-changes   # allow protected (closed, final-data) storms to change
   python -m src.update_catalogue --cnn              # also run CNN inference on the new period

Method: detection is deterministic, so the whole cleaned series is re-detected (cheap) and the result is DIFFED
against the stored catalogue:  unchanged | new | changed | removed.
 * Closed storms built from FINAL data are PROTECTED: if the recomputation alters/removes one, the update aborts
   (unless --accept-changes) because it means upstream data were revised or the code/config changed.
 * Storms with 'open_end' or built from provisional/real-time data may legitimately change; they are replaced and logged.
 * Duplicates are impossible by construction (unique non-overlapping IDs) and re-checked by the validator.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np, pandas as pd
from .common import load_config, resolve
from .data_cleaning import build_raw, clean_series, read_csv_series, validate_clean
from .storm_detection import Params, detect_storms
from .catalogue_validation import assert_valid
from .plotting import plot_catalogue

CMP = ["start_time", "peak_time", "end_time", "min_dst_nt", "intensity_tier", "recovery_to_150_hours", "data_quality_flag"]


def diff_catalogues(old: pd.DataFrame, new: pd.DataFrame):
    o, n = old.set_index("storm_id"), new.set_index("storm_id")
    new_ids = sorted(set(n.index) - set(o.index)); removed = sorted(set(o.index) - set(n.index))
    changed, same = [], []
    for sid in sorted(set(n.index) & set(o.index)):
        a, b = o.loc[sid, CMP], n.loc[sid, CMP]
        eq = all((pd.isna(a[k]) and pd.isna(b[k])) or str(a[k]) == str(b[k]) for k in CMP)
        (same if eq else changed).append(sid)
    return {"new": new_ids, "removed": removed, "changed": changed, "unchanged": same}


def protected(row) -> bool:
    return ("open_end" not in str(row["data_quality_flag"])) and row.get("data_product") == "dst_final"


def run_update(cfg, skip_download=False, since=None, csv=None, product="dst_provisional", accept_changes=False, cnn=False):
    log = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if csv:
        # merge: existing raw rows + new CSV rows
        raw_p = resolve(cfg, "raw_series")
        new_raw = read_csv_series(csv, product, cfg["data"]["missing_value"])
        raw = pd.concat([pd.read_csv(raw_p, parse_dates=["timestamp_utc"]), new_raw], ignore_index=True) if raw_p.exists() else new_raw
        raw = raw.drop_duplicates(["timestamp_utc", "product", "dst_raw"])
    else:
        if not skip_download:
            from . import data_download
            data_download.main((["--since", since] if since else []))
        raw = build_raw(cfg)
    resolve(cfg, "raw_series").parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(resolve(cfg, "raw_series"), index=False)
    clean, rep = clean_series(raw, cfg)
    probs = validate_clean(clean, cfg)
    if probs:
        raise SystemExit("CLEAN DATA VALIDATION FAILED: " + "; ".join(probs))
    clean.to_csv(resolve(cfg, "clean_series"), index=False)
    log["data"] = {k: rep[k] for k in ("first_timestamp", "last_timestamp", "missing_hours", "n_gaps", "hours_by_product")}
    ser = clean.set_index(pd.to_datetime(clean.timestamp_utc))["dst_nt"]; prod = clean.set_index(pd.to_datetime(clean.timestamp_utc))["product"]
    p = Params(**{**Params.from_config(cfg).__dict__, "tiers": cfg["detection"]["tiers"]})
    new = detect_storms(ser, p, prod)
    assert_valid(new, ser, p)
    cat_p = resolve(cfg, "catalogue")
    if cat_p.exists():
        old = pd.read_csv(cat_p, parse_dates=["start_time", "peak_time", "end_time"])
        d = diff_catalogues(old, new)
        lost = [sid for sid in d["changed"] + d["removed"] if protected(old.set_index("storm_id").loc[sid])]
        if lost and not accept_changes:
            raise SystemExit(f"PROTECTED storms would change (final-data, closed): {lost[:10]} ... use --accept-changes after review")
    else:
        d = {"new": new.storm_id.tolist(), "removed": [], "changed": [], "unchanged": []}
    cat_p.parent.mkdir(parents=True, exist_ok=True)
    if cat_p.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); (cat_p.parent / "history").mkdir(exist_ok=True)
        cat_p.replace(cat_p.parent / "history" / f"storm_catalogue_{stamp}.csv")
    new.to_csv(cat_p, index=False)
    todo = set(d["new"]) | set(d["changed"])
    drawn = plot_catalogue(new, ser, resolve(cfg, "plots_dir"), p, only_ids=None)   # manifest hash => only new/changed redrawn
    log["catalogue"] = {k: (len(v) if k == "unchanged" else v) for k, v in d.items()}
    log["n_storms"] = len(new); log["plots_redrawn"] = len(drawn)
    with open(cat_p.parent / "update_log.jsonl", "a") as fh:
        fh.write(json.dumps(log, default=str) + "\n")
    print(json.dumps(log, indent=1, default=str))
    if cnn:
        from .evaluate_cnn import load_model, infer
        m, _ = load_model(cfg)
        first = (pd.Timestamp(rep["last_timestamp"]) - pd.Timedelta(days=62)).strftime("%Y-%m-%d")
        infer(cfg, m, ser, first)
    return new, d


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--since"); ap.add_argument("--csv"); ap.add_argument("--product", default="dst_provisional")
    ap.add_argument("--accept-changes", action="store_true"); ap.add_argument("--cnn", action="store_true")
    a = ap.parse_args(argv)
    run_update(load_config(a.config), a.skip_download, a.since, a.csv, a.product, a.accept_changes, a.cnn)


if __name__ == "__main__":
    main()
