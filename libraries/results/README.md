# torchtomo against LEAP, torch-radon, ASTRA and TIGRE

Measured on one RTX 2080 Ti, torch 2.4.0+cu121, with
`libraries/compare_libraries.py`, on torchtomo `main` at `f0da637`, 2026-09-30.
ASTRA is the `astra-toolbox` 2.5.0 wheel; TIGRE is 3.1.3 at `6b0951a`, built with
`libraries/patches/tigre-texture-copy-sync.patch` (why in `libraries/README.md`).
Each library projects the same phantom and reconstructs its own sinogram, so the
scale each one works in cancels and nothing is corrected by a fitted factor. All
of them get the same angle list, the same phantom, and the same inscribed circle
to be scored over.

Raw numbers are in `library-comparison.json`. `library-comparison.png` shows the
reconstructions and their error maps, `library-summary.png` the four dimensions
side by side.

## The operators agree

Sinograms match to between 0.05% and 0.3% in relative L2 across every size, angle
count and phantom tested. The scale each library needs to reach torchtomo's is
exact and explainable:

| Library | Scale to torchtomo | Relative L2, 256 / 512 px | Why |
| --- | --- | --- | --- |
| LEAP | 1.00000 | 0.20% / 0.10% | same convention once the arc is negated |
| torch-radon | `2 / size` exactly | 0.09% at 512 px | it sums pixel values where torchtomo integrates over a pixel width |
| ASTRA | 0.99996 | 0.30% / 0.15% | same once the arc is negated |
| TIGRE | 1.00000 | 0.12% / 0.07% | same once the arc is negated, started a quarter turn on, and the detector read the other way |

This is a stronger check than the library had before: the torch-radon comparison
could only match up to a fitted ratio, while LEAP, ASTRA and TIGRE are three
independent CUDA implementations that each confirm the absolute scale. TIGRE is
the closest of the four.

## Reconstruction quality, Shepp-Logan

PSNR in dB and SSIM, over the visible circle. Measured with the Ram-Lak ramp that
landed in 0.3; the row below the table is what the same runs gave before it.

| Geometry | torchtomo | torchtomo-cuda | LEAP | torch-radon | ASTRA | TIGRE |
| --- | --- | --- | --- | --- | --- | --- |
| 256 px, 90 angles | 25.59 / 0.648 | 25.59 / 0.648 | 25.55 / 0.618 | 25.59 / 0.648 | 25.55 / 0.618 | 25.60 / 0.650 |
| 256 px, 180 angles | 27.07 / 0.850 | 27.07 / 0.850 | 27.33 / 0.819 | 27.07 / 0.850 | 27.35 / 0.820 | 27.08 / 0.852 |
| 256 px, 360 angles | 27.46 / 0.937 | 27.46 / 0.937 | 27.89 / 0.923 | 27.46 / 0.937 | 27.91 / 0.924 | 27.47 / 0.939 |
| 512 px, 90 angles | 24.48 / 0.534 | 24.48 / 0.534 | 24.20 / 0.511 | 24.48 / 0.534 | 24.23 / 0.511 | 24.49 / 0.535 |
| 512 px, 180 angles | 28.48 / 0.745 | 28.48 / 0.745 | 28.39 / 0.714 | 28.48 / 0.745 | 28.42 / 0.714 | 28.49 / 0.747 |
| 512 px, 360 angles | 30.08 / 0.909 | 30.08 / 0.909 | 30.35 / 0.886 | 30.08 / 0.909 | 30.37 / 0.887 | 30.09 / 0.910 |

The six split into two families by filter. torchtomo, torch-radon and TIGRE all
build Kak and Slaney's spatial ramp kernel and transform it (TIGRE's `ramp_flat`),
and agree to within 0.01 dB and 0.002 of SSIM. LEAP and ASTRA agree with each other
to within 0.03 dB and 0.001. Between the families, torchtomo leads on SSIM in every
row and on PSNR at 90 views and at 512 px with 180, and trails by 0.26 to 0.45 dB
in the other three rows.
The `cuda` backend reproduces the PyTorch path to the same two decimals; its own
kernels are the only difference.

Before the ramp fix the same table read 24.27 / 0.539 at 256 px and 90 views up to
28.78 / 0.682 at 512 px and 360 views, last in every row and by 0.2 of SSIM at the
bottom of it. Both causes were in the FBP filter rather than in the projector, and
both were the same one bin.

## The DC bin was the whole story

The ramp was built analytically as `freq.abs()`, so the DC bin of the padded
transform was exactly zero and the mean of every projection was discarded. The
reconstruction then sat a constant -0.0172 below the phantom, where LEAP's and
torch-radon's biases were zero to five decimal places: the error maps in the
`library-comparison.png` of that time show torchtomo's interior uniformly blue and
the other two white.

torch-radon and skimage instead build Kak and Slaney's kernel (Chapter 3,
Equation 61) in the spatial domain and transform it, which leaves a small positive
DC term, 2 / (pi^2 M) for a transform of length M. Normalised by their maxima the
two filters agree bin for bin to within 3% and differ most at that one bin,
0.000396 against 0.

torchtomo's `src/torchtomo/filters.py` now builds the ramp the second way, for every window on
top of it. The reconstruction bias at 512 px and 360 views is +0.00004, and the
quality table above is the rest of the effect.

## FBP inverts its own forward projector

Reprojecting a reconstruction should return the measurements it came from. That
test used to fail in a way a constant could not fix: torchtomo plateaued at a 10%
residual whatever the angle count, two thirds of it a fixed 9.4% amplitude deficit,
while LEAP converged to 0.45%. Dividing by the fitted gain halved the residual and
cost PSNR, which said the deficit was not a scale.

It was the missing DC bin, which carries about a third of a sinogram's energy. With
the ramp built from the spatial kernel, gain and residual land on LEAP's, from
`libraries/fbp_consistency.py`, 512 px, Shepp-Logan, gain / residual / bias:

| Angles | torchtomo | LEAP |
| ---: | ---: | ---: |
| 45 | 1.0018 / 0.0611 / +0.00001 | 1.0018 / 0.0638 / +0.00001 |
| 90 | 1.0000 / 0.0173 / +0.00001 | 1.0001 / 0.0187 / +0.00000 |
| 180 | 1.0002 / 0.0054 / +0.00007 | 0.9998 / 0.0056 / -0.00000 |
| 360 | 0.9999 / 0.0049 / +0.00004 | 0.9997 / 0.0045 / -0.00000 |
| 720 | 0.9998 / 0.0051 / +0.00002 | 0.9997 / 0.0048 / +0.00000 |

Residual is `||A fbp(y) - y|| / ||y||` and gain is the least squares scale between
the two. torchtomo is ahead at 45 and 90 views and level from 180 on. This is what
any method whose loss round-trips through `fbp()` and `forward()` was paying for:
BM3D, RED, Noise2Inverse and Proj2Proj in `training/`.

Fan beam had the same defect and less of it, because a detector 1.5x wider than the
image leaves less of the projection mean in the DC bin: gain 0.9753 -> 0.9991,
residual 0.0263 -> 0.0049, bias -0.00446 -> -0.00010 at 512 px and 360 views.


## Speed

Milliseconds per call and peak memory PyTorch allocated, 512 px, Shepp-Logan,
remeasured on 2026-09-30. `torchtomo` is the PyTorch path (`backend="torch"`),
`torchtomo-cuda` the runtime-compiled kernels (`backend="cuda"`, the default
where they load). ASTRA and TIGRE show time only; the GPU memory section says why.
Batch 4:

| Operation | Angles | torchtomo | torchtomo-cuda | LEAP | torch-radon | ASTRA | TIGRE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 90 | 3.27 ms, 139 MB | 0.58 ms, 5 MB | 1.11 ms, 6 MB | 0.23 ms, 1 MB | 1.67 ms | 4.39 ms |
| forward | 360 | 17.88 ms, 175 MB | 2.20 ms, 7 MB | 2.63 ms, 11 MB | 0.67 ms, 3 MB | 2.92 ms | 9.65 ms |
| backproject | 90 | 8.20 ms, 80 MB | 0.48 ms, 5 MB | 0.61 ms, 8 MB | 0.16 ms, 4 MB | 1.58 ms | 4.39 ms |
| backproject | 360 | 37.28 ms, 109 MB | 1.98 ms, 7 MB | 1.63 ms, 8 MB | 0.63 ms, 4 MB | 3.40 ms | 7.40 ms |
| fbp | 90 | 3.09 ms, 173 MB | 0.23 ms, 7 MB | 3.44 ms, 8 MB | 0.22 ms, 10 MB | 4.10 ms | 10.80 ms |
| fbp | 360 | 14.31 ms, 175 MB | 0.64 ms, 30 MB | 7.52 ms, 11 MB | 0.88 ms, 44 MB | 6.72 ms | 21.97 ms |

Batch 1:

| Operation | Angles | torchtomo | torchtomo-cuda | LEAP | torch-radon | ASTRA | TIGRE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 90 | 0.91 ms | 0.31 ms | 0.55 ms | 0.06 ms | 1.25 ms | 2.37 ms |
| forward | 360 | 8.16 ms | 1.17 ms | 0.91 ms | 0.20 ms | 1.61 ms | 6.22 ms |
| backproject | 90 | 2.73 ms | 0.46 ms | 0.40 ms | 0.05 ms | 1.10 ms | 1.83 ms |
| backproject | 360 | 15.55 ms | 1.89 ms | 0.85 ms | 0.16 ms | 1.45 ms | 3.43 ms |
| fbp | 90 | 1.76 ms | 0.21 ms | 2.88 ms | 0.18 ms | 1.09 ms | 4.29 ms |
| fbp | 360 | 8.55 ms | 0.28 ms | 5.07 ms | 0.22 ms | 1.75 ms | 12.16 ms |

On its kernels torchtomo is ahead of LEAP on the forward projection at batch 4
(1.2x at 360 views, 1.9x at 90) and at batch 1 with 90 views, and on FBP
everywhere, 12x to 18x. LEAP is ahead on the adjoint at batch 1 (1.2x at 90
views, 2.2x at 360) and at batch 4 with 360 views (1.2x), and on the forward at
batch 1 with 360 views (1.3x). LEAP carries a fixed cost of a few
milliseconds per FBP call, part of it the copy `libraries/leap_projector.py` makes
before handing the sinogram over, which is why its FBP trails even at batch 1.

ASTRA runs on the torch tensors in place, through its batched 3D projector, so
its rows are its kernels and nothing else. torchtomo's kernels are ahead of it on
the forward (1.3x to 4.0x), on the adjoint in three of four rows (1.7x to 3.3x at
batch 4, 2.4x at batch 1 with 90 views), and on FBP by 5x to 18x. ASTRA is ahead
on the adjoint at batch 1 with 360 views, 1.45 against 1.89 ms. Its FBP is the
`FBP_CUDA` algorithm, one image at a time.

TIGRE is the slowest of the CUDA libraries, 2x to 47x behind torchtomo's kernels.
Part of that is its interface rather than its kernels: every call takes NumPy
arrays, so a TIGRE row includes copying the batch to the card and the result
back, and it allocates its working buffers afresh each call.

torch-radon is the fastest projector, 2.5x to 12x ahead of torchtomo's kernels
on forward and adjoint. It samples through the GPU's texture units, whose 8-bit
interpolation weights are fast but not exact, and its adjoint is a pixel-driven
backprojection rather than the transpose of its forward. torchtomo's
`approximate=True` makes the same trade and closes most of that gap (0.99 / 0.45
ms against 0.67 / 0.62 at 360 views, batch 4, in `libraries/README.md`); its
default keeps the forward and adjoint an exact transpose pair, which is what an
unrolled method such as Learned Primal-Dual trains through. On FBP the two are
within 0.06 ms of each other except at batch 4 with 360 views, where torchtomo is
ahead, 0.64 against 0.88 ms; torch-radon is ahead at batch 1.

The PyTorch path runs on any device and in float64, and is 3x to 31x slower than
the kernels. Every comparison of it with LEAP or torch-radon in earlier versions
of this file measured that path; the kernels did not exist yet.

## GPU memory

Peak for a single call, 512 px, 360 angles, batch 4, from the table above:

| Operation | torchtomo | torchtomo-cuda | LEAP | torch-radon |
| --- | ---: | ---: | ---: | ---: |
| forward | 174.7 MB | 7.1 MB | 10.9 MB | 2.9 MB |
| backproject | 109.2 MB | 7.1 MB | 8.4 MB | 4.2 MB |
| fbp | 174.9 MB | 29.5 MB | 10.9 MB | 43.7 MB |

The kernels keep only per-view or per-ray tables on the device, so they sit with
the two CUDA libraries. The PyTorch path holds sampling grids and is the one that
needs hundreds of megabytes.

LEAP allocates outside PyTorch's caching allocator, so `max_memory_allocated`
cannot see all of it. Both a torch-level and a driver-level figure are recorded in
the JSON, and they agree closely here.

ASTRA and TIGRE are left out of this table because neither figure measures them.
Both allocate their textures and working buffers with `cudaMalloc` inside the call
and free them before it returns, so the torch-level peak shows only the tensors
the adapter hands them (5 to 13 MB, recorded in the JSON) and the driver-level
difference before and after the call is zero.

## Fan beam

`python libraries/compare_libraries.py --geometry fan --output libraries/results/fan`
records the same comparison in fan beam, in `fan/`. Quality, Shepp-Logan, PSNR in
dB and SSIM:

| Geometry | torchtomo | torchtomo-cuda | LEAP | torch-radon | ASTRA | TIGRE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 px, 90 angles | 22.51 / 0.543 | 22.51 / 0.543 | 22.46 / 0.523 | 22.86 / 0.552 | 22.37 / 0.520 | 22.52 / 0.543 |
| 256 px, 180 angles | 26.47 / 0.740 | 26.47 / 0.740 | 26.84 / 0.717 | 26.49 / 0.752 | 26.76 / 0.709 | 26.61 / 0.740 |
| 256 px, 360 angles | 27.44 / 0.889 | 27.44 / 0.889 | 28.00 / 0.872 | 27.26 / 0.893 | 27.97 / 0.865 | 27.55 / 0.890 |
| 512 px, 90 angles | 20.89 / 0.483 | 20.89 / 0.483 | 20.77 / 0.465 | 21.34 / 0.491 | 20.70 / 0.463 | 20.90 / 0.485 |
| 512 px, 180 angles | 26.25 / 0.624 | 26.25 / 0.624 | 26.22 / 0.604 | 26.48 / 0.635 | 26.11 / 0.600 | 26.28 / 0.625 |
| 512 px, 360 angles | 29.38 / 0.812 | 29.38 / 0.812 | 29.61 / 0.794 | 29.26 / 0.823 | 29.51 / 0.786 | 29.39 / 0.813 |

LEAP's, ASTRA's and TIGRE's fan-beam sinograms each agree with torchtomo's to
0.6% at 512 px and 1.1% at 256 px, at every view count, with a fitted scale
between 1.0003 and 1.0007. Three independent implementations landing at the same
distance suggests it is torchtomo's fan discretisation rather than a disagreement
about the geometry; the three were not compared with each other. torch-radon's are 35% apart after scaling: its fan geometry is
parametrised differently and `compare_libraries.py` does not map it onto
torchtomo's, so its fan row compares reconstructions of what each library was
asked for, not the same scan.

Speed and memory, 512 px, batch 4:

| Operation | Angles | torchtomo | torchtomo-cuda | LEAP | torch-radon | ASTRA | TIGRE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 90 | 4.03 ms, 131 MB | 0.72 ms, 5 MB | 1.91 ms, 9 MB | 0.42 ms, 1 MB | 2.27 ms | 21.77 ms |
| forward | 360 | 16.06 ms, 134 MB | 2.74 ms, 9 MB | 4.91 ms, 13 MB | 1.43 ms, 4 MB | 3.75 ms | 55.63 ms |
| backproject | 90 | 12.96 ms, 44 MB | 0.73 ms, 6 MB | 1.10 ms, 9 MB | 0.21 ms, 4 MB | 3.58 ms | 10.33 ms |
| backproject | 360 | 51.78 ms, 44 MB | 3.00 ms, 13 MB | 3.52 ms, 9 MB | 0.81 ms, 4 MB | 12.31 ms | 18.17 ms |
| fbp | 90 | 3.76 ms, 146 MB | 0.30 ms, 17 MB | 5.98 ms, 13 MB | 0.31 ms, 22 MB | 6.19 ms | 37.46 ms |
| fbp | 360 | 14.90 ms, 151 MB | 1.10 ms, 63 MB | 18.09 ms, 13 MB | 1.30 ms, 83 MB | 16.05 ms | 79.72 ms |

In fan beam torchtomo's kernels are ahead of LEAP on every operation, 1.8x to
2.7x on the forward, 1.2x to 1.5x on the adjoint and 16x to 20x on FBP, and ahead
of torch-radon on FBP at 360 views. They are ahead of ASTRA at batch 4 on every
operation too, 1.4x to 3.2x on the forward, 4.1x to 4.9x on the adjoint and 15x to
21x on FBP; ASTRA has no batched fan projector, so the adapter calls its 2D one
per image. At batch 1 with 360 views ASTRA's forward is the faster, 1.01 against
1.35 ms (`fan/library-comparison.json`). TIGRE runs fan beam as a one-row cone,
one image per call with the copies each call makes, and is 6x to 125x behind.

## History: when this comparison ran on the PyTorch path

Before 0.3 had its CUDA kernels, this file compared LEAP and torch-radon with the
PyTorch path alone, and most of the gap was not the price of pure PyTorch: at 512
px, 360 angles, batch 4, more time went to `expand().reshape()` copying tensors
(0.630 ms per chunk) than to `grid_sample` doing the work (0.485 ms). Removing that
duplication and building the sampling grids on demand took the forward from 29.94
to 17.88 ms and the adjoint's peak from 4709 MB to 615 MB, which is what let LPD
at 512 px with `--lpd-iterations 10 --lpd-width 32` train at batch 5 on an 11 GB
card. The adjoint has since moved to grid_sample's input backward (37.5 ms, 109
MB above). The file used to end on the conclusion that closing the rest of the
gap meant writing ray-driven CUDA kernels; torchtomo now has them, which is the
`torchtomo-cuda` column.

## What it cost the benchmark: ten methods on each backend

`training/train.py --projector leap` swaps the kernels and changes nothing else,
so the same ten methods, schedules and seeds run on LEAP's operators. Real CT,
512 px, 90 angles, 100k photons, in-window PSNR, before and after torchtomo 0.3
replaced the sampled ramp with the Ram-Lak kernel:

| Method | torchtomo, old ramp | torchtomo 0.3 | LEAP | 0.3 - LEAP |
| --- | ---: | ---: | ---: | ---: |
| FBP | 12.60 | 12.45 | 12.25 | +0.20 |
| SIRT | 19.99 | 19.99 | 19.96 | +0.03 |
| SART | 20.09 | 20.09 | 20.03 | +0.06 |
| iRadonMAP | 23.33 | 23.54 | 23.20 | +0.34 |
| FBP + U-Net | 25.67 | 25.95 | 25.89 | +0.06 |
| LPD | 24.31 | 23.58 | 23.47 | +0.11 |
| RED | 24.37 | 25.65 | 25.57 | +0.08 |
| Noise2Inverse | 18.66 | 20.96 | 20.86 | +0.10 |
| FBP + BM3D | 19.90 | 24.43 | 24.66 | -0.23 |
| Proj2Proj | 12.73 | 23.41 | 23.19 | +0.22 |
| noiseless FBP reference | 20.09 | 24.37 | 24.04 | +0.33 |

The old column is `results-ctw-cuda` as it was recorded before the fix; the new
one is the same run repeated on 0.3, same 100k photons, so the two differ in the
filter alone. Ellipse phantoms at 512 px moved the same way: Proj2Proj 22.49 to
36.68 dB over the circle, from 12.95 dB behind LEAP to 0.66.

The reading that produced the fix was that the gap followed the operators rather
than the methods. SIRT, SART and iRadonMAP matched within 0.05 dB because they
never call `fbp()`, using only the matched `forward()` and `adjoint()` pair the
two libraries agree on, while everything routed through `fbp()` trailed, most of
all the one method whose training loss round-trips through both. Proj2Proj
perturbs a sinogram, reconstructs it, denoises, forward projects the result and
compares with the measurements, so it had to absorb a 10% round-trip
inconsistency before it could denoise anything, from a masked loss that sees one
sixteenth of the entries.

Removing that inconsistency removed the gap, which is as close to a proof of the
mechanism as this benchmark can give: the same ten methods on the same data, one
filter bin apart. What is left is BM3D at 0.23 dB and, in fan beam, Proj2Proj at
1.03 dB, the only rows where LEAP is still ahead by more than a tenth.

Fan beam had less to gain throughout, because a detector 1.5x wider than the
image leaves less of the projection mean in the DC bin: on CT its noiseless
reference moves 20.55 to 20.71 dB where parallel beam moves 20.09 to 24.37.
`training/results-512-fan-leap` and `training/results-ctw-fan-leap` are LEAP's
fan-beam runs, recorded alongside these.


## The FBP baseline is the one untrained method whose hyperparameter is never searched

`classical.py` selects SIRT's iteration count, SART's relaxation, BM3D's sigma and
RED's prior weight on the validation split, and records each search. FBP's filter
is not among them: `fbp()` is called with its default `"ramp"` everywhere. On noisy
data that is the sharpest and noisiest choice available, and it is worth several
decibels.

torchtomo's own filters on the real CT slices, selected on validation exactly as
the other untrained methods are:

| Filter | Val circle | Val window | Test circle | Test window |
| --- | ---: | ---: | ---: | ---: |
| ramp (current, fixed) | 23.470 | 12.905 | 23.715 | 12.585 |
| cosine | 27.877 | 15.863 | 28.322 | 15.481 |
| shepp-logan | 28.468 | 16.304 | 28.953 | 15.937 |
| hamming | 29.157 | 16.818 | 29.693 | 16.479 |
| **hann** | **29.509** | **17.087** | **30.080** | **16.775** |

Hann wins on validation and carries over to test. Selecting the filter the way
every other untrained method is selected moves the FBP row from **12.59 dB to
16.78 dB in window**, and from 23.72 to 30.08 over the circle. That reorders the
table: FBP stops being far below Proj2Proj and closes much of the distance to SIRT
and SART.

The same effect appears in LEAP, whose ramp filter order is also tunable. On the
same slices its order 0 filter reaches 17.85 dB in window against 12.26 for
Ram-Lak, a 5.6 dB spread from one parameter.

This is worth fixing before the table is published, for two reasons. The
comparison currently flatters every method that post-processes an FBP image,
because it hands them the worst available starting point, and it understates the
analytic baseline the learned methods are being measured against.

The filter rows above were measured on the sampled ramp. torchtomo 0.3's ramp
already recovers most of what the ramp row was losing here, 12.60 to 12.45 dB in
window on noisy CT but 20.09 to 24.37 dB on the noiseless reference, so the
spread between filters is worth remeasuring before the search is added.

## Reproducing

```bash
# quality and figures, tolerant of a busy card
python libraries/compare_libraries.py --sections quality figure summary

# speed and memory, which need the card to themselves
python libraries/compare_libraries.py --sections performance

# the same in fan beam
python libraries/compare_libraries.py --geometry fan --output libraries/results/fan
```
