import argparse

import numpy as np

from src.data import common_mask
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    dirs = [f"{d['cache']}/train", f"{d['cache']}/test"]

    common = common_mask(dirs)
    dst = f"{d['cache']}/common_mask.npy"
    np.save(dst, common)

    n_valid = int(common.sum())
    print(f"{dst}: valid {n_valid}, masked {common.size - n_valid}")
    for frac in cfg["sensors"]["fracs"]:
        print(f"  {frac:.3%} -> {max(1, round(frac * n_valid))} sensors")


if __name__ == "__main__":
    main()
