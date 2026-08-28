import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/v0_1.csv")
    ap.add_argument("--out", default="figures/error_vs_fraction.png")
    args = ap.parse_args()

    acc = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["field"] == "mean":
                acc[(r["method"], float(r["frac"]))].append(float(r["err"]))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for method in sorted({k[0] for k in acc}):
        fracs = sorted(k[1] for k in acc if k[0] == method)
        med = [np.median(acc[(method, f)]) for f in fracs]
        lo = [np.percentile(acc[(method, f)], 25) for f in fracs]
        hi = [np.percentile(acc[(method, f)], 75) for f in fracs]
        ax.errorbar(fracs, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)],
                    marker="o", capsize=3, label=method)

    ax.set_xscale("log")
    ax.set_xlabel("sensor fraction")
    ax.set_ylabel("relative L2 error")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
