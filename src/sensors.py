import numpy as np


def place(mask, fracs, rng):
    """Nested sensor sets on the valid points of mask. Returns {frac: flat idx array}."""
    valid = np.flatnonzero(mask.ravel())
    if valid.size == 0:
        raise ValueError("mask has no valid points; the cached case is empty")
    perm = rng.permutation(valid)

    # One permutation, prefixes of decreasing length: the 0.1% set is a subset of the
    # 1.5% set is a subset of the 5% set. Without nesting, part of the difference
    # between fractions is placement luck rather than sensor count.
    out = {}
    for frac in sorted(fracs, reverse=True):
        k = max(1, int(round(frac * valid.size)))
        out[frac] = np.sort(perm[:k])
    return out


# The decoder needs input element k to mean one fixed location, and QR produces
# one global sensor set by construction. Neither can use per-case placement, so
# sensors are placed once over the common mask with a single RNG. The nesting
# rule is the same as place, so the body is shared.
def place_global(common, fracs, rng):
    """Nested sensor sets placed once on the common mask, shared by every case."""
    return place(common, fracs, rng)


def to_mask(idx, shape):
    """Boolean (n, n) mask that is True at the given flat sensor indices."""
    m = np.zeros(int(np.prod(shape)), dtype=bool)
    m[idx] = True
    return m.reshape(shape)


def coords(idx, n):
    """(k, 2) row/col coordinates for flat sensor indices on an n x n grid."""
    return np.column_stack(np.unravel_index(idx, (n, n)))
