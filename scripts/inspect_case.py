import os
import sys

import airfrans as af

from src.utils import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
root = cfg["data"]["root"]

names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
print(f"cases: {len(names)}")
print(f"first: {names[0]}")

sim = af.Simulation(root=root, name=names[0])
pos = sim.position
print(f"points: {pos.shape[0]}")
print(f"x range: {pos[:, 0].min():.3f} {pos[:, 0].max():.3f}")
print(f"y range: {pos[:, 1].min():.3f} {pos[:, 1].max():.3f}")
print(f"point_data keys: {list(sim.internal.point_data.keys())}")
