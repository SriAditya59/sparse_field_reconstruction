import random
import zlib
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed):
    """Seed the global python and numpy RNGs."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path="configs/default.yaml"):
    """Read the YAML config into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def case_rng(seed, name):
    # Per-case RNG keyed on the case name so sensor sets do not depend on iteration
    # order. crc32, not hash(): builtin string hashing is salted per process.
    return np.random.default_rng(seed + zlib.crc32(name.encode()))


def cache_paths(cache_dir):
    return sorted(Path(cache_dir).glob("*.npz"))
