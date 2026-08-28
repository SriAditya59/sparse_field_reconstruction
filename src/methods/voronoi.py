import numpy as np
from scipy.spatial import cKDTree

from src import sensors


def reconstruct(fields, mask, idx):
    """Nearest-sensor fill. Every grid point takes the value of the closest sensor."""
    n = mask.shape[0]
    pts = sensors.coords(idx, n)
    tree = cKDTree(pts)

    rows, cols = np.mgrid[0:n, 0:n]
    query = np.column_stack([rows.ravel(), cols.ravel()])

    # Distance is Euclidean in grid space and ignores the airfoil, so a point below
    # the trailing edge can be assigned a sensor from the suction side. Fixing that
    # needs a geodesic distance and would stop this being the trivial baseline.
    _, nn = tree.query(query, k=1)

    flat = fields.reshape(3, -1)
    out = flat[:, idx[nn]].reshape(3, n, n)
    out[:, ~mask] = 0.0
    return out
