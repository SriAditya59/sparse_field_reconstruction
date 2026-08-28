import argparse
from pathlib import Path

import numpy as np

from src.utils import load_config


def one_case(n, box, rng):
    x = np.linspace(box[0], box[1], n)
    y = np.linspace(box[2], box[3], n)
    X, Y = np.meshgrid(x, y)

    aoa = np.deg2rad(rng.uniform(-5, 15))
    t = rng.uniform(0.08, 0.18)
    Xr = X * np.cos(aoa) + Y * np.sin(aoa)
    Yr = -X * np.sin(aoa) + Y * np.cos(aoa)

    body = (Xr > 0) & (Xr < 1) & (np.abs(Yr) < 5 * t * (0.29 * np.sqrt(np.clip(Xr, 0, 1)) - 0.13 * Xr - 0.35 * Xr**2 + 0.28 * Xr**3 - 0.10 * Xr**4))
    mask = ~body

    r = np.sqrt(np.clip(Xr - 0.25, -9, 9) ** 2 + Yr**2) + 0.05
    circ = rng.uniform(0.4, 1.2)
    u = 1.0 + circ * Yr / r**2 * 0.1
    v = -circ * (Xr - 0.25) / r**2 * 0.1
    wake = np.exp(-((Yr) ** 2) / 0.005) * (Xr > 1) * 0.4
    u = u - wake
    p = -0.5 * (u**2 + v**2 - 1.0)

    fields = np.stack([u, v, p]).astype(np.float32)
    fields[:, ~mask] = 0.0
    return fields, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cases", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    out = Path(d["cache"]) / "test"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    for i in range(args.cases):
        fields, mask = one_case(d["n"], d["box"], rng)
        np.savez_compressed(out / f"synthetic_{i:03d}.npz", fields=fields, mask=mask)
    print(f"{args.cases} synthetic cases -> {out}")


if __name__ == "__main__":
    main()
