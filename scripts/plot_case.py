import argparse

import matplotlib.pyplot as plt
import numpy as np

from src import sensors
from src.data import FIELDS, load_case
from src.utils import cache_paths, case_rng, load_config


def show(ax, arr, mask, box, title):
    im = ax.imshow(np.where(mask, arr, np.nan), origin="lower", extent=box, cmap="RdBu_r")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--frac", type=float, default=0.001)
    ap.add_argument("--out", default="figures/ground_truth.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = cache_paths(f'{cfg["data"]["cache"]}/{args.split}')[0]
    fields, mask = load_case(path)
    box = cfg["data"]["box"]

    idx = sensors.place(mask, [args.frac], case_rng(cfg["seed"], path.stem))[args.frac]
    pts = sensors.coords(idx, mask.shape[0])
    x = box[0] + pts[:, 1] * (box[1] - box[0]) / (mask.shape[0] - 1)
    y = box[2] + pts[:, 0] * (box[3] - box[2]) / (mask.shape[0] - 1)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for j, (f, lab) in enumerate(zip(fields, FIELDS)):
        im = show(axes[0, j], f, mask, box, lab)
        fig.colorbar(im, ax=axes[0, j], shrink=0.8)
        show(axes[1, j], np.full_like(f, np.nan), mask, box, f"{lab}, {len(idx)} sensors")
        axes[1, j].scatter(x, y, c=f[pts[:, 0], pts[:, 1]], s=18, cmap="RdBu_r", edgecolors="k", linewidths=0.3)

    fig.suptitle(path.stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
