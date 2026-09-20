# torchtomo-benchmark

Benchmarks for [torchtomo](https://github.com/itu-biai/torchtomo): how it compares
with other CT projectors, and how reconstruction methods built on it compare with
each other, with every recorded result. Everything that needs a library torchtomo
does not depend on is here; torchtomo's own speed and self-consistency checks,
which run on torch alone, stay in its `benchmark/`.

| Folder | What it holds |
| --- | --- |
| [`libraries/`](libraries/README.md) | torchtomo against LEAP, torch-radon, and scikit-image: operator agreement, reconstruction quality, speed, and GPU memory. The recorded comparison is in `libraries/results/`. |
| [`training/`](training/README.md) | One pipeline that scores ten methods on ellipse phantoms or real CT slices: FBP, SIRT, SART, FBP+BM3D, RED, FBP+U-Net, iRadonMAP, Noise2Inverse, Proj2Proj, and Learned Primal-Dual. Recorded runs are in `training/results*/`. |
| `tests/` | Operator agreement with LEAP, torch-radon, and scikit-image, and checks on the training code. |

## Setup

```bash
git clone git@github.com:itu-biai/torchtomo-benchmark.git
cd torchtomo-benchmark
python -m venv .venv
.venv/bin/pip install -e /path/to/torchtomo    # or the release: pip install torchtomo
.venv/bin/pip install -r requirements.txt
make test
```

The recorded runs use features that are not released yet, such as
`backend="cuda"`, so install torchtomo from a checkout. LEAP and torch-radon are
optional; [`libraries/README.md`](libraries/README.md) has LEAP's build steps.
Tests and scripts that need either one skip when it is missing.

Everything runs from the repository root:

```bash
python training/train.py --help
python training/train.py --image-size 512 --angles 90 --batch-size 5 --device cuda \
    --backend cuda --output training/results-512-cuda
python libraries/compare_libraries.py
python libraries/speed_table.py
```

## Recorded runs

All at 512 x 512 with 90 angles except the first. The tables and figures are in
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
