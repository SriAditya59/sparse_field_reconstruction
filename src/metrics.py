import numpy as np


def eval_mask(mask, sens):
    """Points the error is measured on: valid, and not handed to the method."""
    return mask & ~sens


def rel_l2(pred, true, mask, sens):
    """Relative L2 error per field, on non-sensor non-masked points. Returns (3,) array."""
    ev = eval_mask(mask, sens)
    d = pred[:, ev] - true[:, ev]
    return np.linalg.norm(d, axis=1) / np.linalg.norm(true[:, ev], axis=1)


def median_iqr(errs):
    """Median and (q25, q75) across cases for a 1D array of errors."""
    errs = np.asarray(errs)
    return np.median(errs), np.percentile(errs, 25), np.percentile(errs, 75)
