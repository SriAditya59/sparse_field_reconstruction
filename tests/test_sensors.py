import numpy as np

from src import sensors
from src.utils import case_rng


def make_mask(n=64):
    m = np.ones((n, n), dtype=bool)
    m[28:36, 20:44] = False
    return m


def test_sets_are_nested():
    mask = make_mask()
    sets = sensors.place(mask, [0.05, 0.015, 0.001], case_rng(0, "a"))
    big, mid, small = sets[0.05], sets[0.015], sets[0.001]
    assert set(small) <= set(mid) <= set(big)


def test_sensors_avoid_mask_and_are_reproducible():
    mask = make_mask()
    a = sensors.place(mask, [0.05], case_rng(0, "case_x"))[0.05]
    b = sensors.place(mask, [0.05], case_rng(0, "case_x"))[0.05]

    assert np.array_equal(a, b)
    assert mask.ravel()[a].all()
    assert len(a) == round(0.05 * mask.sum())
