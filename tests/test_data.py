import numpy as np

from src.data import common_mask


def test_common_mask_is_and_over_cases(tmp_path):
    a = np.ones((8, 8), dtype=bool)
    a[2, 2] = False
    b = np.ones((8, 8), dtype=bool)
    b[5, 5] = False
    for name, m in [("a", a), ("b", b)]:
        np.savez_compressed(
            tmp_path / f"{name}.npz", fields=np.zeros((3, 8, 8), np.float32), mask=m
        )

    common = common_mask([tmp_path])
    assert common.sum() == 62
    assert not common[2, 2]
    assert not common[5, 5]
    assert common[0, 0]
