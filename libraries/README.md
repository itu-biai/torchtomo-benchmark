# Library comparisons

torchtomo against LEAP, torch-radon, ASTRA, TIGRE, and scikit-image. All but
scikit-image are optional: every script and test here skips what is not installed.

| File | What it does |
| --- | --- |
| `compare_libraries.py` | torchtomo against LEAP, torch-radon, ASTRA, and TIGRE: PSNR, SSIM, speed, GPU memory, and a figure |
| `speed_table.py` | forward, adjoint, and FBP in milliseconds for every torchtomo backend and every other library, in one process |
| `throughput.py` | slices per second for torchtomo, scikit-image, and torch-radon |
| `accuracy_vs_skimage.py` | reconstruction quality on analytic phantoms, against scikit-image |
| `fbp_consistency.py` | whether a library's FBP inverts its own forward projector: gain, residual, bias |
| `visual_comparison.py` | torchtomo and scikit-image reconstructions side by side at 512 px |
| `visualize.py` | parallel- and fan-beam reconstructions with their PSNR and SSIM |
| `compare_geometry_gradients.py` | the cost of a geometry gradient against Thies et al.'s differentiable backprojector; writes `results/geometry-gradients.json` |
| `leap_projector.py` | `LeapParallelBeam` and `LeapFanBeam`, drop-ins for torchtomo's projectors backed by LEAP's kernels |
| `astra_projector.py` | `AstraParallelBeam` and `AstraFanBeam`, the same for ASTRA, on the torch tensors in place through DLPack |
| `tigre_projector.py` | `TigreParallelBeam` and `TigreFanBeam`, the same for TIGRE |
| `matched_pair.py` | the autograd the drop-ins share: each library's forward and backprojection as each other's gradient |
| `results/` | the recorded comparison: `README.md`, `library-comparison.json` and its figures, and the same in fan beam in `results/fan/` |

The operator agreement tests are `tests/test_leap_consistency.py`,
`tests/test_torchradon_consistency.py`, `tests/test_astra_tigre_consistency.py`,
and `tests/test_skimage_consistency.py`.
`training/leap_compare.py` scores LEAP's own FBP, SIRT, and SART on the data a
training run wrote.

## Speed, one process

512 x 512, batch 4, RTX 2080 Ti, milliseconds for forward / adjoint / FBP,
from `speed_table.py` on torchtomo `f0da637`, 2026-09-30, all in one run, whose
rows are in `results/speed-table.json`:

| Geometry, angles | torchtomo `torch` | torchtomo `cuda` | torchtomo `cuda`, approximate | LEAP | torch-radon | ASTRA | TIGRE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parallel, 360 | 17.9 / 37.7 / 14.3 | 2.2 / 2.0 / 0.6 | 1.0 / 0.4 / 0.6 | 2.6 / 1.7 / 7.5 | 0.7 / 0.6 / 0.9 | 2.9 / 3.5 / 7.0 | 9.8 / 7.2 / 22.7 |
| parallel, 90 | 3.3 / 8.3 / 3.1 | 0.6 / 0.5 / 0.2 | 0.3 / 0.1 / 0.2 | 1.1 / 0.6 / 3.5 | 0.2 / 0.2 / 0.2 | 1.7 / 1.6 / 4.2 | 4.2 / 3.6 / 8.0 |
| fan, 360 | 16.1 / 52.3 / 14.9 | 2.8 / 3.1 / 1.1 | 1.3 / 0.8 / 1.1 | 5.0 / 3.6 / 17.2 | 1.4 / 0.8 / 1.3 | 3.7 / 12.3 / 16.0 | 56.1 / 18.3 / 78.8 |
| fan, 90 | 4.8 / 13.8 / 3.8 | 0.7 / 0.7 / 0.3 | 0.3 / 0.2 / 0.3 | 1.9 / 1.1 / 6.0 | 0.4 / 0.2 / 0.3 | 2.3 / 3.6 / 6.8 | 18.9 / 10.3 / 37.0 |

torch-radon samples through the GPU's texture units, whose 8-bit interpolation
weights are fast but not exact, and its adjoint is a pixel-driven backprojection
rather than the transpose of its forward. torchtomo's `approximate=True` makes the
same trade; its default `cuda` backend keeps the forward and adjoint an exact
transpose pair.

ASTRA works on the torch tensors in place; TIGRE's rows include copying the batch
to the card and back, which its NumPy interface makes every call pay. Neither has a
batched fan-beam projector, so both run fan beam one image at a time.

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

## ASTRA

```bash
pip install astra-toolbox
```

The PyPI wheel (2.5.0) carries its own CUDA runtime. On the benchmark machine it
was installed with `pip install --target` into a directory on `PYTHONPATH`, with
the NumPy and SciPy it pulled in removed, so the venv's torch kept its own.

`astra_projector.py` hands ASTRA the torch tensors themselves through DLPack
(`astra.experimental.direct_FP3D` and friends, `astra.data2d.link`), so no call
copies anything to the host. Parallel beam goes through ASTRA's 3D parallel
projector with one detector row per image, the batched route tomosipo takes; fan
beam through its 2D fan projector one image at a time, since a 3D cone would mix
stacked slices. FBP is ASTRA's own `FBP_CUDA` with its Ram-Lak filter.

ASTRA turns the gantry the other way round, and its 2D volumes run bottom to top.
With those mapped and the image and detector on [-1, 1], its sinograms agree with
torchtomo's to 0.3% in parallel beam and 1.1% in fan beam at 256 px and 180
views, with no fitted scale.

## TIGRE

```bash
git clone https://github.com/CERN/TIGRE.git
git -C TIGRE apply ../torchtomo-benchmark/libraries/patches/tigre-texture-copy-sync.patch
pip install --no-deps ./TIGRE        # builds the CUDA extensions with the system nvcc
```

TIGRE 3.1.3 (`6b0951a`) built with nvcc 11.5 as it stands, but its forward
projectors have a race that the patch closes. They copy the image into a CUDA
array with `cudaMemcpy3DAsync` and launch the projection kernels on a different
stream without waiting, and TIGRE builds with `--default-stream=per-thread`, so
nothing orders the two. The first block of views can then read a half-copied
image. Unpatched, a parallel-beam projection at 512 px and 180 views came back
with its first 8 views at about 62% of their value in 8 of 18 runs, depending on
what else sat on the card; one recorded fan-beam run lost 8% of its agreement
with torchtomo the same way. With one `cudaDeviceSynchronize()` after each copy
it was right in 18 of 18, in both geometries. The backprojectors already wait on
their copy stream. Its interface is NumPy, so each
call copies its input to the card and its result back; that is what a TIGRE user
pays and it stays in the timings.

A 2D problem is a TIGRE volume one voxel thick. Parallel beam stacks the batch as
slices; fan beam is a cone with one detector row, run one image at a time. Two
details matter. TIGRE's interpolated projector samples the whole source to
detector segment, so a parallel "source" placed far away costs samples in
proportion (a million units away made a single projection take minutes); two
units clears the image. And its `"matched"` backprojector is not scaled as the
adjoint of its own `Ax`: the dot-product test is off by exactly the pixel width in
parallel beam and by about 0.49 in fan beam. The adapter measures that ratio once
on TIGRE's own `Ax` and multiplies by it, so the pair it trains with is matched.

TIGRE starts its gantry a quarter turn on from torchtomo's and reads its detector
the other way; in parallel beam it also turns the other way round. Its sinograms
then agree with torchtomo's to 0.12% in parallel beam and 1.1% in fan beam, with
no fitted scale.

## Running the whole comparison

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
figure from `torch.cuda.mem_get_info` is the one that covers every library.
ASTRA and TIGRE also allocate their working buffers outside it.
