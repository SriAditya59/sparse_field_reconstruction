import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src import metrics, sensors
from src.data import FIELDS, load_case
from src.methods import voronoi
from src.utils import case_rng, cache_paths, load_config, set_seed

METHODS = {"voronoi": voronoi.reconstruct}


def sweep(cfg, methods, split="test", limit=None):
    """Run every method at every sensor fraction over the cached cases of one split."""
    cache = f"{cfg['data']['cache']}/{split}"
    paths = cache_paths(cache)[:limit]
    if not paths:
        raise SystemExit(f"no cached cases in {cache}")

    rows = []
    for path in paths:
        name = path.stem
        fields, mask = load_case(path)
        sets = sensors.place(mask, cfg["sensors"]["fracs"], case_rng(cfg["seed"], name))

        for frac, idx in sets.items():
            sens = sensors.to_mask(idx, mask.shape)
            for mname in methods:
                t = time.perf_counter()
                pred = METHODS[mname](fields, mask, idx)
                dt = time.perf_counter() - t

                err = metrics.rel_l2(pred, fields, mask, sens)
                for f, e in zip(FIELDS, err):
                    rows.append((name, mname, frac, len(idx), f, e, dt))
                rows.append((name, mname, frac, len(idx), "mean", err.mean(), dt))
    return rows


def summarise(rows):
    keys = sorted({(r[1], r[2]) for r in rows}, key=lambda k: (k[0], -k[1]))
    print(f"{'method':10s} {'frac':>7s} {'n_sens':>7s} {'median':>8s} {'q25':>8s} {'q75':>8s}")
    for mname, frac in keys:
        sel = [r for r in rows if r[1] == mname and r[2] == frac and r[4] == "mean"]
        med, q25, q75 = metrics.median_iqr([r[5] for r in sel])
        print(f"{mname:10s} {frac:7.3%} {sel[0][3]:7d} {med:8.4f} {q25:8.4f} {q75:8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", nargs="+", default=["voronoi"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/v0_1.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    rows = sweep(cfg, args.methods, args.split, args.limit)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method", "frac", "n_sens", "field", "err", "time_s"])
        w.writerows(rows)

    print(f"{len(rows)} rows -> {args.out}\n")
    summarise(rows)


if __name__ == "__main__":
    main()
