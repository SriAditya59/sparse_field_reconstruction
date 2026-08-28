import os
import sys

import numpy as np

from src.utils import cache_paths, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
d = cfg["data"]

paths = cache_paths(d["cache"])
print(f"cached npz files: {len(paths)}")
if paths:
    frac = []
    for p in paths[:20]:
        m = np.load(p)["mask"]
        frac.append(m.sum() / m.size)
    print(f"valid fraction over first {len(frac)} cached cases: "
          f"min {min(frac):.4f} max {max(frac):.4f}")

import airfrans as af

names = sorted(x for x in os.listdir(d["root"]) if os.path.isdir(os.path.join(d["root"], x)))
sim = af.Simulation(root=d["root"], name=names[0])
b = sim.internal.bounds
print(f"case: {names[0]}")
print(f"mesh x {b[0]:.3f} {b[1]:.3f}")
print(f"mesh y {b[2]:.3f} {b[3]:.3f}")
print(f"mesh z {b[4]:.6f} {b[5]:.6f}")
print(f"config box {d['box']}")
print(f"point_data keys {list(sim.internal.point_data.keys())}")
