import argparse
import time

from src.data import build_cache, list_names, make_grid, mesh_z, sample_case
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    names = list_names(d["root"], d["task"], train=args.split == "train")[: args.limit]
    cache = f"{d['cache']}/{args.split}"

    if args.time_only:
        import airfrans as af

        sim = af.Simulation(root=d["root"], name=names[0])
        grid = make_grid(d["box"], d["n"], mesh_z(sim))
        t = time.perf_counter()
        fields, mask = sample_case(sim, grid, d["n"], d["u_key"], d["p_key"])
        dt = time.perf_counter() - t
        n_valid = int(mask.sum())
        print(f"{dt:.2f} s/case, {len(names)} cases -> {dt * len(names) / 60:.1f} min")
        print(f"valid {n_valid}, masked {mask.size - n_valid}")
        for frac in cfg["sensors"]["fracs"]:
            print(f"  {frac:.3%} -> {max(1, round(frac * n_valid))} sensors")
        return

    print(f"{args.split}: {len(names)} cases -> {cache}")
    build_cache(d["root"], cache, names, d["box"], d["n"], d["u_key"], d["p_key"])


if __name__ == "__main__":
    main()
