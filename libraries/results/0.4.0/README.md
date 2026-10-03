# torchtomo 0.4.0 benchmark refresh

Measured on NVIDIA GeForce RTX 2080 Ti, torch 2.4.0+cu121, CUDA 12.1. torchtomo release **0.4.0**, local source commit `0c5f70039fa8ddb29303ae23943a9abe1486e98b`. The JSON records source hashes, package versions, settings and section timestamps.

The PyTorch and CUDA columns explicitly select their backends. Each library reconstructs its own sinogram without a fitted reconstruction gain. PSNR uses the visible circle; SSIM averages the 7×7 SSIM map over valid window centres inside that circle. The SSIM values therefore use a different region from the historical tables. TIGRE was unavailable in this environment.

## Parallel beam

[Raw measurements](parallel/library-comparison.json), [reconstructions](parallel/library-comparison.png), [summary figure](parallel/library-summary.png).

Shepp–Logan: circle PSNR (dB) / circle SSIM.

| Size, views | torchtomo | torchtomo-cuda | leap | torch-radon | astra |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 px, 90 views | 25.59 / 0.580 | 25.59 / 0.580 | 25.55 / 0.544 | 25.59 / 0.580 | 25.55 / 0.543 |
| 256 px, 180 views | 27.07 / 0.819 | 27.07 / 0.819 | 27.33 / 0.781 | 27.07 / 0.818 | 27.35 / 0.782 |
| 256 px, 360 views | 27.46 / 0.924 | 27.46 / 0.924 | 27.89 / 0.906 | 27.46 / 0.923 | 27.91 / 0.907 |
| 512 px, 90 views | 24.48 / 0.430 | 24.48 / 0.430 | 24.20 / 0.401 | 24.48 / 0.430 | 24.23 / 0.401 |
| 512 px, 180 views | 28.48 / 0.687 | 28.48 / 0.687 | 28.39 / 0.648 | 28.48 / 0.686 | 28.42 / 0.648 |
| 512 px, 360 views | 30.08 / 0.887 | 30.08 / 0.887 | 30.35 / 0.859 | 30.08 / 0.886 | 30.37 / 0.860 |

Milliseconds per call, 512 px, batch 4; mean over 20 calls after three warmups.

| Operation, views | torchtomo | torchtomo-cuda | leap | torch-radon | astra |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward, 90 | 3.268 | 0.580 | 1.129 | 0.226 | 1.697 |
| forward, 360 | 17.852 | 2.196 | 2.616 | 0.668 | 2.936 |
| backproject, 90 | 8.171 | 0.480 | 0.622 | 0.162 | 1.609 |
| backproject, 360 | 37.426 | 1.963 | 1.631 | 0.627 | 3.417 |
| fbp, 90 | 3.097 | 0.241 | 3.463 | 0.217 | 4.237 |
| fbp, 360 | 14.292 | 0.635 | 7.598 | 0.878 | 6.957 |

## Fan beam

[Raw measurements](fan/library-comparison.json), [reconstructions](fan/library-comparison.png), [summary figure](fan/library-summary.png).

Shepp–Logan: circle PSNR (dB) / circle SSIM.

| Size, views | torchtomo | torchtomo-cuda | leap | torch-radon | astra |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 px, 90 views | 22.51 / 0.468 | 22.51 / 0.468 | 22.46 / 0.444 | 22.86 / 0.477 | 22.37 / 0.440 |
| 256 px, 180 views | 26.47 / 0.690 | 26.47 / 0.690 | 26.84 / 0.661 | 26.49 / 0.704 | 26.76 / 0.653 |
| 256 px, 360 views | 27.44 / 0.866 | 27.44 / 0.866 | 28.00 / 0.844 | 27.26 / 0.870 | 27.97 / 0.836 |
| 512 px, 90 views | 20.89 / 0.371 | 20.89 / 0.371 | 20.77 / 0.349 | 21.34 / 0.380 | 20.70 / 0.347 |
| 512 px, 180 views | 26.25 / 0.540 | 26.25 / 0.540 | 26.22 / 0.515 | 26.48 / 0.553 | 26.11 / 0.511 |
| 512 px, 360 views | 29.38 / 0.769 | 29.38 / 0.769 | 29.61 / 0.746 | 29.26 / 0.782 | 29.51 / 0.737 |

Milliseconds per call, 512 px, batch 4; mean over 20 calls after three warmups.

| Operation, views | torchtomo | torchtomo-cuda | leap | torch-radon | astra |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward, 90 | 4.026 | 0.713 | 1.893 | 0.418 | 2.275 |
| forward, 360 | 16.089 | 2.720 | 4.892 | 1.422 | 3.866 |
| backproject, 90 | 12.922 | 0.726 | 1.092 | 0.210 | 3.592 |
| backproject, 360 | 50.901 | 2.983 | 3.516 | 0.803 | 12.368 |
| fbp, 90 | 3.759 | 0.301 | 6.000 | 0.315 | 6.210 |
| fbp, 360 | 14.891 | 1.095 | 17.779 | 1.295 | 16.037 |

The torch-radon input arc is now negated to match this scan. Its fitted-scale relative L2 disagreement is 0.61%–1.11% across these sizes/views, instead of approximately 35% in the historical comparison. A regression test also checks the physical `2 / size` pixel-width scale without fitting it.

## GPU memory scope

The memory panels show torchtomo only. `torch_peak_mb` is the additional PyTorch allocation peak above the warmed baseline, not total device memory. For LEAP, torch-radon, ASTRA and TIGRE, external allocations are unmeasured; their PyTorch counters are partial observations. `driver_delta_mb` measures retained memory before/after a call and misses buffers freed inside it. No total-memory ranking is inferred from those values.

## Real CT: validation-tuned FBP

512×512, 90 views, fixed 100,000 photons/ray, dataset seed 2026. The packed CT data use 200 training, 50 validation and 50 test slices; patient groups are disjoint. Filters are selected on validation window PSNR only, then evaluated once on test. Values below are mean per-slice PSNR in the display window and visible circle. This is a baseline rerun, not a retraining of the learned reconstruction methods.

| Geometry | Backend | Ramp test (dB) | Tuned test (dB) | Selected filter | Validation (dB) |
| --- | --- | ---: | ---: | --- | ---: |
| parallel | [torchtomo](../../../training/results-ct-fbp-0.4.0-torchtomo-parallel/metrics.json) | 12.45 | 17.56 | hann | 17.74 |
| parallel | [leap](../../../training/results-ct-fbp-0.4.0-leap-parallel/metrics.json) | 12.23 | 17.86 | leap-order-0 | 18.03 |
| fan | [torchtomo](../../../training/results-ct-fbp-0.4.0-torchtomo-fan/metrics.json) | 12.17 | 16.67 | hann | 16.75 |
| fan | [leap](../../../training/results-ct-fbp-0.4.0-leap-fan/metrics.json) | 12.08 | 16.99 | leap-order-0 | 17.04 |

torchtomo searches ramp, Shepp–Logan, cosine, Hamming and Hann. LEAP searches native orders 12, 0 and 2; the candidate families are different and are recorded by name. The tuned baseline is additional to ramp FBP. Learned methods retain their ramp inputs.

## Training validation

Three training seeds (2026, 2027, 2028), fixed image/split/noise seed 2026 and fixed dose, were run on both torchtomo CUDA and LEAP with all reconstruction methods plus tuned FBP. These **32×32, two-epoch smoke runs** validate execution, checkpoint handling and repeated-seed bookkeeping. They do not establish convergence or a real-CT method ranking. [Repeated-seed summary](../../../training/smoke-0.4.0-summary.json) reports the sample standard deviation of test means across training runs, separately from slice variation.

The historical full training results have not been relabeled as 0.4.0. A new learned-method ranking needs full schedules with corrected masks and matched doses across several training seeds.

## Reproduce

Install `requirements.txt` (torchtomo pinned at 0.4.0), or install the matching local checkout with `pip install -e ../torchtomo`. Then:

```bash
python libraries/compare_libraries.py --sections quality performance figure summary
python libraries/compare_libraries.py --geometry fan --sections quality performance figure summary
python training/train.py --device cuda --backend cuda --image-size 512 \
    --data-dir training/ct-subset --models "" --classical-methods fbp-tuned \
    --output training/results-ct-fbp-0.4.0-torchtomo-parallel
```

Use `--geometry fan` and/or `--projector leap` with a separate output directory for the other CT baseline rows. See [the training protocol](../../../training/README.md) for repeated training seeds and dose calibration. Timing requires an idle GPU.
