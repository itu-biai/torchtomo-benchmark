# Library comparisons

torchtomo against LEAP, torch-radon, and scikit-image. LEAP and torch-radon are
optional: every script and test here skips what is not installed.

| File | What it does |
| --- | --- |
| `compare_libraries.py` | torchtomo against LEAP and torch-radon: PSNR, SSIM, speed, GPU memory, and a figure |
| `speed_table.py` | forward, adjoint, and FBP in milliseconds for every torchtomo backend, LEAP, and torch-radon, in one process |
| `throughput.py` | slices per second for torchtomo, scikit-image, and torch-radon |
| `accuracy_vs_skimage.py` | reconstruction quality on analytic phantoms, against scikit-image |
| `fbp_consistency.py` | whether a library's FBP inverts its own forward projector: gain, residual, bias |
| `visual_comparison.py` | torchtomo and scikit-image reconstructions side by side at 512 px |
| `visualize.py` | parallel- and fan-beam reconstructions with their PSNR and SSIM |
| `compare_geometry_gradients.py` | the cost of a geometry gradient against Thies et al.'s differentiable backprojector; writes `results/geometry-gradients.json` |
| `leap_projector.py` | `LeapParallelBeam` and `LeapFanBeam`, drop-ins for torchtomo's projectors backed by LEAP's kernels |
| `results/` | the recorded comparison: `README.md`, `library-comparison.json` and its figures, and the same in fan beam in `results/fan/` |

The operator agreement tests are `tests/test_leap_consistency.py`,
`tests/test_torchradon_consistency.py`, and `tests/test_skimage_consistency.py`.
`training/leap_compare.py` scores LEAP's own FBP, SIRT, and SART on the data a
training run wrote.

## Speed, one process

512 x 512, batch 4, RTX 2080 Ti, milliseconds for forward / adjoint / FBP,
from `speed_table.py`, remeasured on 2026-09-30 within timing noise of these:

| Geometry, angles | torchtomo `torch` | torchtomo `cuda` | torchtomo `cuda`, approximate | LEAP | torch-radon |
| --- | --- | --- | --- | --- | --- |
| parallel, 360 | 18.0 / 38.3 / 14.3 | 2.3 / 2.1 / 0.6 | 1.0 / 0.5 / 0.6 | 2.7 / 1.7 / 7.8 | 0.7 / 0.6 / 0.9 |
| parallel, 90 | 3.3 / 8.4 / 3.1 | 0.6 / 0.5 / 0.2 | 0.3 / 0.1 / 0.2 | 1.1 / 0.6 / 3.5 | 0.2 / 0.2 / 0.2 |
| fan, 360 | 16.2 / 53.0 / 14.9 | 2.8 / 3.2 / 1.1 | 1.3 / 0.8 / 1.1 | 5.0 / 3.6 / 17.5 | 2.0 / 1.1 / 1.6 |
| fan, 90 | 4.1 / 13.1 / 3.8 | 0.7 / 0.8 / 0.3 | 0.4 / 0.2 / 0.3 | 2.0 / 1.1 / 6.1 | 0.4 / 0.2 / 0.3 |

torch-radon samples through the GPU's texture units, whose 8-bit interpolation
weights are fast but not exact, and its adjoint is a pixel-driven backprojection
rather than the transpose of its forward. torchtomo's `approximate=True` makes the
same trade; its default `cuda` backend keeps the forward and adjoint an exact
transpose pair.

## Geometry gradients, against Thies et al.

[geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)
(Thies et al., Phys. Med. Biol. 2023) differentiates a numba CUDA fan-beam
backprojector with respect to each view's projection matrix. It has no forward
projector. `compare_geometry_gradients.py` times one operation plus the gradient
of a scalar loss with respect to a per-view lateral translation, batch of one,
median of 10, RTX 2080 Ti; torchtomo's pose table is on its `main`, not
released yet.

| Size, views, bins | Thies et al. backprojection | torchtomo forward | torchtomo adjoint | torchtomo backproject |
| --- | --- | --- | --- | --- |
| 256, 360, 384 | 62.4 ms, 19 MiB | 4.4 ms, 46 MiB | 6.2 ms, 46 MiB | 2.8 ms, 19 MiB |
| 512, 360, 768 | 246 ms, 23 MiB | 5.3 ms, 76 MiB | 6.7 ms, 76 MiB | 4.0 ms, 24 MiB |
| 512, 720, 768 | 275 ms, 27 MiB | 9.5 ms, 133 MiB | 12.6 ms, 133 MiB | 7.2 ms, 25 MiB |

torchtomo carries the geometry gradient on its CUDA kernels for all three: the
forward and adjoint are 14 to 46 times faster than Thies et al.'s backprojector,
and its own FBP backprojection 22 to 62 times, in the same memory. Before
torchtomo#5 that backprojection differentiated the geometry on the PyTorch path,
which keeps every sampling grid for the backward: 92 and 365 ms in 3.2 and 9.0 GB
at the first two sizes, and out of memory at the third.

```bash
git clone https://github.com/mareikethies/geometry_gradients_CT ~/geometry_gradients_CT
pip install "numba-cuda[cu13]"   # [cu12] on a CUDA 12 driver
python libraries/compare_geometry_gradients.py --thies ~/geometry_gradients_CT
```

numba-cuda 0.30 still imports `np.row_stack`, which NumPy 2.4 removed; the
script puts the alias back.

## LEAP

[LEAP](https://github.com/LLNL/LEAP) (LivermorE AI Projector, LLNL) is an
independent CUDA implementation, so it checks the parallel-beam geometry including
its absolute scale: with the image and detector both on `[-1, 1]`, the two forward
projectors agree without any fitted factor. LEAP turns the gantry the other way
round, which is the one convention `leap_projector.py` negates.

It is not on PyPI, so it is built from source. On CUDA 11.5 with GCC 11 the build
needs two changes, both in `src/CMakeLists.txt`:

```bash
git clone --depth 1 https://github.com/LLNL/LEAP.git && cd LEAP
sed -i 's/find_package(CUDA 11.7 REQUIRED)/find_package(CUDA 11.5 REQUIRED)/' src/CMakeLists.txt
# all-major builds every architecture; pin your own to keep the build short
sed -i 's/CUDA_ARCHITECTURES all-major/CUDA_ARCHITECTURES 75/' src/CMakeLists.txt
pip install cmake
mkdir -p build && cd build
# C++17 is what breaks: nvcc 11.5 cannot parse GCC 11's <functional> under it
cmake .. -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_STANDARD=14
cmake --build . -j 8
cd .. && pip install . --no-build-isolation
```

`setup.py` recompiles from scratch, so stub `etc/build.sh` with `exit 0` before the
last step to keep the library that was just built.

Two LEAP behaviours matter when comparing:

- its default ramp filter is Shepp-Logan (order 2), not Ram-Lak. `LeapParallelBeam`
  pins order 12, which is what torchtomo's `"ramp"` is.
- its CPU parallel-beam kernel faults on a volume of more than one slice, so
  `LeapParallelBeam` does CPU-resident work on the card and hands the result back.

### Running the comparisons

From the repository root:

```bash
# agreement of the operators
pytest tests/test_leap_consistency.py

# LEAP's own untrained methods, on the data a finished run wrote
python training/leap_compare.py --results training/results-ctw

# the whole ten-method benchmark, on LEAP's kernels instead of torchtomo's
python training/train.py --projector leap --output training/results-ctw-leap ...
```

## torch-radon

```bash
pip install torch-radon
```

torch-radon 1.0 predates two removals it depends on, `np.int` and `torch.rfft`, so
its `filter_sinogram` raises on any modern PyTorch. `compare_libraries.py` restores
both in its own process rather than editing the installed package, and it still
builds the filter with torch-radon's own `construct_fourier_filter`, so only the
transform calls differ from what the library shipped.

Its sinograms are torchtomo's divided by the pixel width: the measured scale is
exactly `2 / size` at every size tested, because torch-radon sums pixel values
where torchtomo integrates over a pixel of physical width.

## Three-way comparison

```bash
python libraries/compare_libraries.py            # writes libraries/results/
python libraries/compare_libraries.py --geometry fan --output libraries/results/fan
```

Each library projects the same phantom and reconstructs its own sinogram, so the
scale each one works in cancels and the quality figures need no fitted correction.
Speed and memory are only meaningful on an idle card; `--sections quality figure`
runs the parts that are not.

GPU memory is reported two ways. LEAP allocates outside PyTorch's caching
allocator, so `torch.cuda.max_memory_allocated` cannot see it; the driver-level
figure from `torch.cuda.mem_get_info` is the one that covers all three.
