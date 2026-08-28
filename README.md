# sparse-field-reconstruction

Reconstructing 2D RANS flow fields around airfoils from a small number of point sensors,
and measuring how the error grows as sensors are removed.

Dataset: [AirfRANS](https://airfrans.readthedocs.io) (`scarce` split). Every case is
interpolated once onto a shared 128x128 grid over `x in [-0.5, 1.5]`, `y in [-1.0, 1.0]`,
with the airfoil interior masked. All methods operate on that grid.

The common grid costs near-wall resolution. Reconstruction on the native unstructured
mesh is a separate piece of work and is not done here.

## Result

Not yet measured on AirfRANS. Numbers below are from the synthetic fixture used to build
the harness and are not a result.

| method | 5% (806 sens) | 1.5% (242) | 0.1% (16) |
|---|---|---|---|
| voronoi | 0.120 | 0.200 | 0.491 |

## Setup

```
pip install -e .
python -c "import airfrans as af; af.dataset.download(root='data/raw', unzip=True, OpenFOAM=False)"
python scripts/build_cache.py
python -m src.run --methods voronoi
```

## Metric

Relative L2 error per field, evaluated only on non-sensor, non-masked points, so no
method is credited for returning values it was given. Reported as median and
interquartile range across test cases.

## Sensor placement

Uniform random over valid grid points, one permutation per case, prefixes taken for each
fraction. The 0.1% set is a subset of the 1.5% set is a subset of the 5% set, so
differences between fractions are not placement luck.
