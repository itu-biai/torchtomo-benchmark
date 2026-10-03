# torchtomo-benchmark

Benchmarks for [torchtomo](https://github.com/itu-biai/torchtomo): how it compares
with other CT projectors, and how reconstruction methods built on it compare with
each other, with every recorded result. Everything that needs a library torchtomo
does not depend on is here; torchtomo's own speed and self-consistency checks,
which run on torch alone, stay in its `benchmark/`.

| Folder | What it holds |
| --- | --- |
| [`libraries/`](libraries/README.md) | torchtomo against LEAP, torch-radon, ASTRA, TIGRE, and scikit-image: operator agreement, reconstruction quality, speed, and GPU memory. The recorded comparison is in `libraries/results/`. |
| [`training/`](training/README.md) | One pipeline that scores ten methods and a tuned FBP baseline on ellipse phantoms or real CT slices: FBP, SIRT, SART, FBP+BM3D, RED, FBP+U-Net, iRadonMAP, Noise2Inverse, Proj2Proj, and Learned Primal-Dual. Recorded runs are in `training/results*/`. |
| [`geometry/`](geometry/README.md) | What differentiating the scan geometry buys: the centre of rotation of measured scans (HTC 2022, a walnut) against Vo's method and phase correlation, and per-view motion and angle errors against projection matching. Uses torchtomo 0.4.0. |
| `tests/` | Operator agreement with LEAP, torch-radon, ASTRA, TIGRE, and scikit-image, and checks on the training code. |

## Setup

```bash
git clone git@github.com:itu-biai/torchtomo-benchmark.git
cd torchtomo-benchmark
python -m venv .venv
.venv/bin/pip install -e ../torchtomo        # local 0.4.0 source; optional with the pinned release
.venv/bin/pip install -r requirements.txt
make test
```

The benchmarks use **torchtomo 0.4.0**, pinned in `requirements.txt`; CUDA kernels
and geometry gradients are included in that release. LEAP, torch-radon, ASTRA,
and TIGRE are optional; [`libraries/README.md`](libraries/README.md) has the
build steps. Tests and scripts that need one skip when it is missing.

Everything runs from the repository root:

```bash
python training/train.py --help
python training/train.py --image-size 512 --angles 90 --batch-size 5 --device cuda \
    --backend cuda --photons 100000 --output training/results-512-0.4.0
python libraries/compare_libraries.py
python libraries/speed_table.py
python geometry/calibrate_real.py
```

## Recorded runs

Fresh 0.4.0 library measurements and corrected metrics are in
[`libraries/results/0.4.0/`](libraries/results/0.4.0/README.md). The training
directories below are historical; they have not been relabeled as 0.4.0 runs.
New training runs add a validation-tuned FBP baseline and use a fixed photon dose.

The historical runs below are at 512 x 512 with 90 angles except the first. The tables and figures are in
[`training/README.md`](training/README.md).

| Directory | Data | Geometry | Projector | Methods |
| --- | --- | --- | --- | --- |
| `results` | ellipses, 64 x 64, CPU | parallel | torchtomo | all ten |
| `results-512` | ellipses | parallel | torchtomo | FBP, U-Net, LPD with 5 iterations |
| `results-512-lpd10` | ellipses | parallel | torchtomo | FBP, U-Net, LPD with 10 iterations and width 32 |
| `results-512-all` | ellipses | parallel | torchtomo | all ten |
| `results-512-all-leap` | ellipses | parallel | LEAP | all ten |
| `results-512-cuda` | ellipses | parallel | torchtomo, `backend="cuda"` | all ten |
| `results-512-fan` | ellipses | fan | torchtomo | all ten |
| `results-512-fan-cuda` | ellipses | fan | torchtomo, `backend="cuda"` | all ten |
| `results-512-fan-leap` | ellipses | fan | LEAP | all ten |
| `results-ctw-six-methods` | CT slices | parallel | torchtomo | FBP, SIRT, SART, U-Net, RED, LPD |
| `results-ctw` | CT slices | parallel | torchtomo | all ten |
| `results-ctw-leap` | CT slices | parallel | LEAP | all ten |
| `results-ctw-cuda` | CT slices | parallel | torchtomo, `backend="cuda"` | all ten |
| `results-ctw-fan` | CT slices | fan | torchtomo | all ten |
| `results-ctw-fan-cuda` | CT slices | fan | torchtomo, `backend="cuda"` | all ten |
| `results-ctw-fan-leap` | CT slices | fan | LEAP | all ten |

Git keeps each run's configuration, metrics, logs, and figures. Checkpoints,
datasets, and reconstructions (`*.pt`) and the packed CT slices
(`training/ct-subset/`, built by `training/pack_ct_subset.py`) stay out of it.

## History

This code moved here from the torchtomo repository at commit `b349f5c`, from
`examples/ellipses/`, `benchmark/`, and `benchmark-results/`. Its history before
the move is in torchtomo.
