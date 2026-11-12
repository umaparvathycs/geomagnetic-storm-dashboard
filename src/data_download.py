"""Download monthly Kyoto WDC Dst pages (HTML) into data/raw/<product>/YYYYMM.html.

Usage:
  python -m src.data_download                     # all configured months (skips existing final months)
  python -m src.data_download --since 2026-06     # only months >= 2026-06
Needs internet. Raw files are NEVER modified after download. A log is written to data/raw/download_log.csv.
"""
from __future__ import annotations
import argparse, csv, time, urllib.request, urllib.error
from datetime import date, datetime
from pathlib import Path
from .common import load_config, resolve


def month_range(start: str, end: str | None):
    y, m = map(int, start.split("-"))
    if end is None:
        t = date.today(); ey, em = t.year, t.month
    else:
        ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "geomag-storm-project/1.0 (academic)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--since", default=None, help="YYYY-MM; only download months >= this")
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    raw_dir = resolve(cfg, "raw_dir")
    base = cfg["data"]["base_url"]
    refetch = int(cfg["data"]["refetch_last_n_months"])
    today = date.today()
    rows = []
    for prod in cfg["data"]["products"]:
        if not prod.get("start"):
            continue
        for y, m in month_range(prod["start"], prod.get("end")):
            tag = f"{y}{m:02d}"
            if a.since and tag < a.since.replace("-", ""):
                continue
            out = raw_dir / prod["name"] / f"{tag}.html"
            age = (today.year - y) * 12 + (today.month - m)
            if out.exists() and (prod["name"] == "dst_final" or age > refetch):
                continue
            url = f"{base}/{prod['name']}/{tag}/index.html"
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = fetch(url)
                out.write_bytes(data)
                status = f"ok:{len(data)}"
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                status = f"FAILED:{e}"
            print(prod["name"], tag, status)
            rows.append([datetime.utcnow().isoformat(timespec="seconds"), prod["name"], tag, url, status])
            time.sleep(a.sleep)
    log = raw_dir / "download_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc_time", "product", "month", "url", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
