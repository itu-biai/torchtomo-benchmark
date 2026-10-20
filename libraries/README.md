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
| `compare_cone.py` | torchtomo's 3D cone beam and breast tomosynthesis against ASTRA and LEAP: agreement, speed, memory, and reconstruction quality; writes `results/cone/` and `results/dbt/` |
| `cone_projectors.py` | `AstraConeBeam` and `LeapConeBeam`, ASTRA's `cone_vec` and LEAP's modular beam driven by torchtomo's source and detector positions |
| `matched_pair.py` | the autograd the drop-ins share: each library's forward and backprojection as each other's gradient |
| `results/` | the recorded comparison: `README.md`, `speed-table.json`, and `library-comparison.json` with its figures per geometry in `results/parallel/` and `results/fan/` |

The operator agreement tests are `tests/test_leap_consistency.py`,
`tests/test_torchradon_consistency.py`, `tests/test_astra_tigre_consistency.py`,
`tests/test_cone_consistency.py`, and `tests/test_skimage_consistency.py`.
`training/leap_compare.py` scores LEAP's own FBP, SIRT, and SART on the data a
training run wrote.

Current 0.4.0 measurements use circle PSNR and SSIM averaged over valid window
centres inside the circle, explicit `torch`/`cuda` backends, and recorded software
versions and source hashes. See [`results/0.4.0/README.md`](results/0.4.0/README.md).
The tables below retain their original pre-release measurements.

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
median of 10, RTX 2080 Ti; the recorded pose-table measurements predate the 0.4.0 release.

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
python libraries/compare_libraries.py                  # writes libraries/results/parallel/
python libraries/compare_libraries.py --geometry fan   # writes libraries/results/fan/
```

Each library projects the same phantom and reconstructs its own sinogram, so the
scale each one works in cancels and the quality figures need no fitted correction.
Speed and memory are only meaningful on an idle card; `--sections quality figure`
runs the parts that are not.

GPU memory is reported two ways. LEAP allocates outside PyTorch's caching
allocator, so `torch.cuda.max_memory_allocated` cannot see it; the driver-level
figure from `torch.cuda.mem_get_info` is the one that covers every library.
ASTRA and TIGRE also allocate their working buffers outside it.

## 3D cone beam and breast tomosynthesis

Browse the [interactive tables and figures](results/cone-dbt-report/report.html)
offline, or view the [SVG figures](results/cone-dbt-report/README.md) on GitHub.

torchtomo's `ConeBeam` and `BreastTomosynthesis` against ASTRA and LEAP, from
`compare_cone.py`, RTX 2080 Ti, PyTorch 2.14.0+cu130, ASTRA 2.5.0, LEAP 1.26,
2026-10-20. `torchtomo` is ConeBeam's CUDA kernels (`backend="cuda"`), which are
newer than torchtomo 0.4.0: the measurements use the `feat/cone-cuda-kernels`
branch, on top of commit `e2ca99c`, on `PYTHONPATH` (source hash `4ca1fd5e9653` in the JSON). The rows are in
`results/cone/cone-comparison.json` and `results/dbt/cone-comparison.json`, with
every table in each folder's `tables.md`; the PyTorch path's rows are in the
`pytorch-path/` folder inside each. TIGRE was not installed on the benchmark
machine.

```bash
export PYTHONPATH=../torchtomo/src:libraries
python libraries/compare_cone.py --geometry cone     # writes libraries/results/cone/
python libraries/compare_cone.py --geometry dbt \
    --victre ../torchtomo/benchmark/dbt/results/victre/input_bin4.npz      # writes libraries/results/dbt/
python libraries/compare_cone.py --geometry cone --cone-cases 32x90 64x180 128x360 \
    --libraries torchtomo torchtomo-pytorch --sections agreement performance \
    --output libraries/results/cone/pytorch-path
pytest tests/test_cone_consistency.py
```

`cone_projectors.py` gives the other two libraries torchtomo's source and
detector element positions, so the three see one geometry. ASTRA gets them as a
`cone_vec` and LEAP as a modular beam. Two scans are measured:

- **cone**: a circular flat panel scan over a full turn, magnification 2, of a 3D
  modified Shepp-Logan phantom. torchtomo takes two samples per voxel along each
  ray.
- **dbt**: the published VICTRE arc (25 views over 50 degrees, stationary
  detector) on the measured projections and VICTRE's own reconstruction of TCIA
  phantom 694607065, which torchtomo's `benchmark/dbt/prepare_victre.py`
  prepares. Bin 4, 8, and 16 detector pixels are 0.34, 0.68, and 1.36 mm, with
  voxels of the same width and 1 mm slices.

LEAP's modular beam needs cubic voxels. The adapter interpolates thicker or
thinner slices linearly onto cubic voxels before projecting, and maps the
backprojection back with the transpose, so its pair stays matched. Its
`libleapct.so` also wants `cudaGetDeviceProperties_v2` from a CUDA 12 runtime,
which a CUDA 13 PyTorch does not load. `load_leap_runtime()` loads the pip
`nvidia-cuda-runtime-cu12` library globally first.

### Agreement

Relative L2 against torchtomo with no fitted scale. The dot test is
`<A x, y> / <x, A^T y>` for random `x` and `y`.

| Case | ASTRA forward | ASTRA adjoint | ASTRA dot test | LEAP forward | LEAP adjoint | LEAP dot test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cone 32³, 90 views | 1.94% | 2.72% | 1.0054 | 2.04% | 2.76% | 1.0628 |
| cone 64³, 180 views | 1.07% | 1.95% | 1.0054 | 0.74% | 1.96% | 1.0319 |
| cone 128³, 360 views | 0.51% | 1.40% | 1.0053 | 0.41% | 1.40% | 1.0159 |
| cone 256³, 360 views | 0.25% | 1.02% | 1.0053 | 0.22% | 1.00% | 1.0079 |
| VICTRE, 1.36 mm | 0.26% | 4.02% | 1.0444 | 1.07% | 15.90% | 1.0308 |
| VICTRE, 0.68 mm | 0.46% | 4.27% | 1.0447 | 0.23% | 7.67% | 1.0378 |
| VICTRE, 0.34 mm | 0.57% | 4.23% | 1.0448 | 0.31% | 4.69% | 1.0413 |

The forward projectors agree to within 2%, and closer as the grid gets finer,
so the conventions and the absolute scale match: the fitted forward scale is
1 to within 0.4%, except LEAP at 1.36 mm (0.991). torchtomo's dot test is 1 to
within 3e-7 in every case, because its adjoint is the exact transpose of its
forward. ASTRA's and LEAP's backprojectors are only approximately matched to
their projectors, off by 0.5% to 6%. In tomosynthesis both backproject about
3.5% less than torchtomo (fitted scale 0.962 to 0.967), which is their 4% dot
test departure. LEAP's 15.9% at 1.36 mm comes from interpolating 1 mm slices
onto 1.36 mm cubic voxels. torchtomo's kernels and its PyTorch path agree to
within 6e-6 relative L2 everywhere (the `pytorch-path` tables).

### Speed and memory

Milliseconds per forward / backprojection of one volume, mean of at least three
calls after a warmup:

| Case | torchtomo | ASTRA | LEAP | torchtomo forward + geometry gradient |
| --- | ---: | ---: | ---: | ---: |
| cone 32³, 90 views | 0.4 / 0.5 | 1.6 / 1.1 | 0.4 / 0.6 | 1.0 |
| cone 64³, 180 views | 2.2 / 2.7 | 2.5 / 1.4 | 1.9 / 1.2 | 4.2 |
| cone 128³, 360 views | 32.1 / 37.4 | 13.9 / 5.2 | 25.7 / 11.2 | 50.1 |
| cone 256³, 360 views | 262 / 332 | 102 / 31.7 | 207 / 82.6 | 397 |
| VICTRE, 1.36 mm | 0.5 / 0.7 | 1.3 / 1.3 | 0.6 / 1.2 | 1.4 |
| VICTRE, 0.68 mm | 1.6 / 2.8 | 1.9 / 1.5 | 1.7 / 6.5 | 3.8 |
| VICTRE, 0.34 mm | 9.2 / 15.5 | 9.9 / 3.2 | 8.7 / 49.6 | 20.8 |

The kernels sample every ray at fixed steps, with exact trilinear weights, at
about 100 million samples per millisecond on this card at every size. A ray
reads the volume from a copy stored in 4 x 4 x 4 voxel bricks with a zero border,
so a sample's eight voxels share one or two cache blocks, and consecutive samples
that share all eight load nothing. In tomosynthesis torchtomo's forward is as fast
as ASTRA's and LEAP's. In the circular scan it is faster up to 64³ and 2.3 to 2.6
times slower than ASTRA from 128³ up: ASTRA steps once per voxel through the
GPU's texture interpolation, where torchtomo takes two exact samples per voxel,
which `n_samples` sets.

The adjoint is the exact transpose, scattered with atomic additions: each ray
sums its consecutive samples' weights per voxel in registers and adds a voxel
once its stencil has moved past it, which makes it about three times faster than
one atomic per sample and voxel. Rays whose value is zero, outside the phantom
here, cost nothing; with random projections the 256³ adjoint takes 0.74 s.
ASTRA's and LEAP's backprojectors gather per voxel, which is faster from 128³ up
(torchtomo takes 7 to 10 times ASTRA's time and 3 to 4 times LEAP's) but is not
their projector's transpose. In tomosynthesis at 0.34 mm torchtomo's adjoint is 3
times faster than LEAP's and 5 times slower than ASTRA's.

The geometry gradient, with respect to every source and detector element
position, comes from a kernel of its own that accumulates, per ray, the sampled
values and the trilinear gradient along it, so it stores nothing per sample. The
other two libraries do not offer one.

| Case | torchtomo, PyTorch path | torchtomo, CUDA kernels |
| --- | ---: | ---: |
| cone 32³, 90 views: forward / adjoint / with geometry gradient | 101 / 106 / 3,966 ms, 4.1 GB | 0.4 / 0.4 / 0.9 ms |
| cone 64³, 180 views | 1,612 / 1,769 ms / out of memory | 2.3 / 2.7 / 4.3 ms |
| cone 128³, 360 views | 25,789 / 27,935 ms / out of memory | 32.3 / 37.6 / 50.5 ms |
| VICTRE, 1.36 mm | 189 / 206 / 7,485 ms, 6.4 GB | 0.5 / 0.7 / 1.4 ms |
| VICTRE, 0.68 mm | 1,494 / 1,629 ms / out of memory | 1.6 / 2.8 / 3.9 ms |
| VICTRE, 0.34 mm | 11,985 / 13,175 ms / out of memory | 9.4 / 15.7 / 21.1 ms |

That is 250 to 1,300 times faster for the operators and 4,400 to 5,300 times for
the geometry gradient where the PyTorch path fits, whose autograd keeps about 300
bytes per sample. The kernels' peak PyTorch allocation for the gradient is 1.6 GB
at 256³ with 360 views, and 0.27 GB for VICTRE at 0.34 mm.

The PyTorch peak memory in the JSON counts only the input and output tensors
for ASTRA and LEAP. Both allocate their working buffers outside PyTorch's
allocator, so those figures are not a memory comparison.

### Reconstruction quality

Every library runs the same reconstruction code with its own operators.

Cone, noiseless projections of each library's own forward projector, 30 SIRT
iterations, PSNR inside the inscribed cylinder:

| Case | torchtomo SIRT | ASTRA SIRT | LEAP SIRT | ASTRA FDK |
| --- | ---: | ---: | ---: | ---: |
| cone 32³, 90 views | 18.79 dB | 18.95 dB | 18.90 dB | 18.52 dB |
| cone 64³, 180 views | 19.27 dB | 19.37 dB | 19.34 dB | 21.77 dB |
| cone 128³, 360 views | 20.10 dB, 3.3 s | 20.15 dB, 0.6 s | 20.13 dB, 1.2 s | 24.94 dB |
| cone 256³, 360 views | 20.60 dB, 29.2 s | 20.62 dB, 4.2 s | 20.61 dB, 9.2 s | 27.82 dB |

torchtomo has no FDK, so that column is ASTRA's `FDK_CUDA` for reference. On the
PyTorch path the 128³ SIRT took 1,667 s.

VICTRE, measured projections. FBP is the Hann-windowed ramp of torchtomo's
`reconstruct_victre.py` followed by each library's backprojector. OS-SART is 10
iterations of 5 subsets fitted to the 13 even views. Both are scored by their
correlation with VICTRE's own reconstruction inside the breast, and OS-SART also
by its relative residual on the 12 held-out odd views:

| Case | FBP corr. (torchtomo / ASTRA / LEAP) | OS-SART corr. | Held-out residual | OS-SART seconds |
| --- | --- | --- | --- | --- |
| VICTRE, 1.36 mm | 0.857 / 0.865 / 0.793 | 0.431 / 0.433 / 0.423 | 3.54% / 3.53% / 3.60% | 0.0 / 0.1 / 0.1 |
| VICTRE, 0.68 mm | 0.884 / 0.892 / 0.879 | 0.449 / 0.455 / 0.454 | 3.57% / 3.59% / 3.62% | 0.1 / 0.1 / 0.1 |
| VICTRE, 0.34 mm | 0.884 / 0.886 / 0.888 | 0.435 / 0.438 / 0.438 | 3.65% / 3.66% / 3.68% | 0.3 / 0.3 / 0.6 |

The three reconstruct equally well: within 0.16 dB in SIRT, and in
tomosynthesis within 0.015 in correlation and 0.1 percentage points in held-out
residual. The exception is LEAP's FBP at 1.36 mm, which carries the slice
interpolation above. OS-SART correlates less with the reference than FBP does
because the reference is itself an FBP.
