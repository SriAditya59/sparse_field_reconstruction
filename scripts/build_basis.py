import numpy as np

from src.data import load_case
from src.utils import cache_paths, load_config


def main():
    cfg = load_config()
    d = cfg["data"]
    common = np.load(f"{d['cache']}/common_mask.npy")
    paths = cache_paths(f"{d['cache']}/train")

    X = np.stack([load_case(p)[0][:, common] for p in paths])  # (n, 3, n_common)
    mean = X.mean(axis=0)                                      # (3, n_common)

    # p is O(1e3) and u, v are O(1e1). Divide each field by its own std or the
    # leading modes are pure pressure with the velocity content as noise on top.
    # This is the silent failure of the phase.
    std = X.std(axis=(0, 2))                                   # (3,)
    A = ((X - mean) / std[:, None]).reshape(len(paths), -1)    # (n, 3*n_common)

    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    ev = np.cumsum(S**2) / np.sum(S**2)

    print(f"{len(paths)} snapshots, {int(common.sum())} points/field, {Vt.shape[0]} modes")
    print(f"  mode 1 alone: {ev[0]:.4f}")
    for k in cfg["pod"]["modes"]:
        print(f"  {k:3d} modes: {ev[k - 1]:.4f}")

    dst = f"{d['cache']}/basis_128.npz"
    np.savez_compressed(dst, modes=Vt, sv=S, mean=mean, std=std, common=common)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
