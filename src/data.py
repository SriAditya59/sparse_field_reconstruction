import json
from pathlib import Path

import numpy as np

FIELDS = ("u", "v", "p")


def make_grid(box, n, z=0.0):
    """Structured n x n grid over box=(x0, x1, y0, y1), placed on the plane z."""
    import pyvista as pv

    x0, x1, y0, y1 = box
    grid = pv.ImageData()
    grid.dimensions = (n, n, 1)
    grid.origin = (x0, y0, z)
    grid.spacing = ((x1 - x0) / (n - 1), (y1 - y0) / (n - 1), 1.0)
    return grid


def mesh_z(sim):
    # AirfRANS internal meshes are planar 2D cells sitting at a single z that is not
    # necessarily 0. Probing from a grid on the wrong plane returns zero valid points.
    z0, z1 = sim.internal.bounds[4:6]
    return 0.5 * (z0 + z1)


def sample_case(sim, grid, n, u_key="U", p_key="p"):
    """Probe one AirfRANS simulation onto the grid. Returns (3, n, n) fields and (n, n) mask."""
    out = grid.sample(sim.internal)

    U = np.asarray(out[u_key]).reshape(n, n, -1)
    p = np.asarray(out[p_key]).reshape(n, n)
    fields = np.stack([U[..., 0], U[..., 1], p]).astype(np.float32)

    # Probe points that hit no cell are the airfoil interior, so the mask falls out of
    # the interpolation instead of needing a separate inside test.
    mask = np.asarray(out["vtkValidPointMask"]).reshape(n, n).astype(bool)
    fields[:, ~mask] = 0.0
    return fields, mask


def build_cache(root, cache, names, box, n, u_key="U", p_key="p"):
    """Interpolate each named case onto the grid and write one npz per case."""
    import airfrans as af

    Path(cache).mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(names):
        dst = Path(cache) / f"{name}.npz"
        if dst.exists():
            continue

        sim = af.Simulation(root=root, name=name)
        grid = make_grid(box, n, mesh_z(sim))
        fields, mask = sample_case(sim, grid, n, u_key, p_key)

        if mask.sum() < 0.2 * mask.size:
            raise SystemExit(
                f"{name}: only {mask.sum()}/{mask.size} grid points hit the mesh. "
                f"box {box}, mesh xy bounds {tuple(round(b, 3) for b in sim.internal.bounds[:4])}"
            )

        np.savez_compressed(dst, fields=fields, mask=mask)
        if i == 0 or (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(names)} {name}")


def load_case(path):
    """Read one cached case. Returns (3, n, n) fields and (n, n) bool mask."""
    d = np.load(path)
    return d["fields"], d["mask"]


def list_names(root, task="scarce", train=True):
    """Case names for one AirfRANS split, without loading any field data."""
    # The scarce task reuses the full task's test set.
    key = ("full" if task == "scarce" and not train else task) + ("_train" if train else "_test")
    with open(Path(root) / "manifest.json") as f:
        return sorted(json.load(f)[key])
