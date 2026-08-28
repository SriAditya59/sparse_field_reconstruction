import numpy as np

from src import metrics, sensors
from src.methods import voronoi
from src.utils import case_rng


def setup(n=64):
    mask = np.ones((n, n), dtype=bool)
    mask[28:36, 20:44] = False
    rng = np.random.default_rng(0)
    fields = rng.standard_normal((3, n, n)).astype(np.float32)
    fields[:, ~mask] = 0.0
    idx = sensors.place(mask, [0.05], case_rng(0, "t"))[0.05]
    return fields, mask, idx


def test_error_ignores_sensor_points():
    fields, mask, idx = setup()
    sens = sensors.to_mask(idx, mask.shape)

    pred = fields.copy()
    pred[:, sens] = 1e6
    assert metrics.rel_l2(pred, fields, mask, sens).max() == 0.0


def test_voronoi_copies_sensor_values_exactly():
    fields, mask, idx = setup()
    pred = voronoi.reconstruct(fields, mask, idx)
    flat_pred, flat_true = pred.reshape(3, -1), fields.reshape(3, -1)
    assert np.allclose(flat_pred[:, idx], flat_true[:, idx])


def test_voronoi_beats_zero_fill_on_a_smooth_field():
    n = 64
    mask = np.ones((n, n), dtype=bool)
    y, x = np.mgrid[0:n, 0:n] / n
    fields = np.stack([np.sin(3 * x), np.cos(3 * y), x + y]).astype(np.float32)
    idx = sensors.place(mask, [0.05], case_rng(0, "s"))[0.05]
    sens = sensors.to_mask(idx, mask.shape)

    err = metrics.rel_l2(voronoi.reconstruct(fields, mask, idx), fields, mask, sens)
    assert err.max() < 0.5
