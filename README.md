# sparse-field-reconstruction

Reconstructing 2D RANS flow fields around airfoils from a small number of point sensors,
and measuring how the error grows as sensors are removed.

![ground truth and sensors](figures/ground_truth.png)

Top row: a full AirfRANS case. Bottom row: the 16 sensor readings a method is given at the
0.1% setting. The gap between those two rows is the problem.

## Status

One method of four. Voronoi is the floor that the others have to beat; gappy POD and a
learned decoder land next, solver-based data assimilation in early September.

## Result

Relative L2 error, median over 200 test cases, interquartile range in brackets. Mean over
`u`, `v`, `p`.

| sensors | fraction | voronoi |
|---|---|---|
| 781 | 5% | 0.098 [0.087, 0.140] |
| 234 | 1.5% | 0.141 [0.125, 0.205] |
| 16 | 0.1% | 0.408 [0.372, 0.557] |

Cutting sensors by 49x, from 781 to 16, costs a factor 4.2 in error. Reconstruction takes
under 0.01 s per case at every fraction.

## Where it loses

At 16 sensors the median error is 0.408, so nearly half of every field is unrecovered, and
the IQR spans 0.372 to 0.557 — the spread across cases is nearly half the error itself. A
single sensor landing near the suction peak or not is worth more than the method.

Nearest-sensor distance is Euclidean in grid space and ignores the airfoil, so a point below
the trailing edge can be assigned a sensor from the suction side, across the body. A
geodesic distance would fix it and would stop this being the trivial baseline.

## Data

[AirfRANS](https://airfrans.readthedocs.io), `scarce` task, 200 test cases. Each case is a
different unstructured mesh, so every case is interpolated once onto a shared 128x128 grid
over `x in [-0.5, 1.5]`, `y in [-1.0, 1.0]`. The mesh sits on the plane z=0.5; the grid
follows it.

Every method is evaluated on one common mask: the 15,623 grid points valid in all 1000
train and test cases, with 761 points masked because at least one airfoil covers them. A
fixed point set is required by gappy POD (fixed basis support), the decoder (fixed output
dimension), and QR pivoting (one global sensor set). It costs the near-wall band that only
some airfoils leave uncovered — which is exactly where reconstruction error concentrates,
so the common-mask numbers above sit below the per-case numbers this repo reported before.

The common grid costs near-wall resolution on top of that. Reconstruction on the native
unstructured mesh is separate work and is not done here.

## Metric

Relative L2 error per field, `||pred - true|| / ||true||`, evaluated only on non-sensor,
non-masked points, so no method is credited for returning values it was given. Median and
IQR across cases rather than mean and standard deviation: a few hard cases dominate the
mean and hide the typical behaviour.

## Sensor placement

Sensors are placed once on the common mask with a single RNG (`global` placement), so every
case is reconstructed from sensors at identical locations. This is required by the learned
decoder, whose input element k must always mean the same point, and by QR pivoting, which
produces one global set by construction. The per-case random protocol is still available as
`--placement per_case` and reproduces the earlier numbers.

Placement is one permutation, prefixes taken for each fraction, so the 0.1% set is a subset
of the 1.5% set is a subset of the 5% set and differences between fractions are sensor count
and not placement luck. Sensor locations are identical across methods at a given fraction.

The global set is redrawn whenever the common mask changes: adding the 600 extra full_train
airfoils dropped 14 points and moved the 1.5% median from 0.196 to 0.141. Absolute errors
are that sensitive to which points are picked, so only compare methods, never runs.

## POD basis

Economy SVD of the 800 full_train snapshots on the common mask, each field centred and
divided by its own training std (461 for `p`, 19 for `u`); without that the basis is
essentially all pressure. Eight modes capture 99.9% of the normalised variance and mode 1
alone captures 64%. The fields are low-rank in Re and angle of attack, so gappy POD is not
mode-count limited except at 0.1% coverage, where modes are capped at the sensor count.

## Reproducing

```
pip install -e .
python -c "import airfrans as af; af.dataset.download(root='data/raw', unzip=True, OpenFOAM=False)"
python scripts/build_cache.py --split test
python scripts/build_cache.py --split train --task full
python scripts/build_common_mask.py
python scripts/build_basis.py
python -m src.run --methods voronoi
python scripts/plot_error.py
```

Download is 9 GB and took 35 minutes. Cache build is 0.11 s per case.

## Licence

MIT.
