**table1a**

| Library | Geometry | Forward (ms) | Backproject (ms) | FBP (ms) | PSNR (dB) | Adjoint defect |
|---|---|---|---|---|---|---|
| torchtomo CUDA | parallel | 2.20 | 1.96 | 0.64 | 30.08 | 4e-10 |
| LEAP | parallel | 2.62 | 1.63 | 7.60 | 30.35 | not measured |
| torch-radon | parallel | 0.67 | 0.63 | 0.88 | 30.08 | 5e-05 |
| ASTRA | parallel | 2.94 | 3.42 | 6.96 | 30.37 | 1e-04 |
| torchtomo CUDA | fan | 2.72 | 2.98 | 1.10 | 29.38 | 3e-10 |
| LEAP | fan | 4.89 | 3.52 | 17.78 | 29.61 | not measured |
| torch-radon | fan | 1.42 | 0.80 | 1.29 | 29.26 | 1e-04 |
| ASTRA | fan | 3.87 | 12.37 | 16.04 | 29.51 | 1e-04 |

**table1b**

| Size, views | Thies et al. backprojection (ms) | torchtomo PyTorch backproject (ms) | torchtomo CUDA backproject (ms) | torchtomo CUDA forward (ms) | torchtomo CUDA adjoint (ms) |
|---|---|---|---|---|---|
| 256, 360 | 63.87 (62.91 to 63.99) | 91.72 (91.67 to 91.72) | 2.12 (2.11 to 2.13) | 4.87 (4.80 to 5.47) | 5.02 (4.36 to 5.50) |
| 512, 360 | 250.98 (248.93 to 251.00) | 361.84 (361.74 to 361.86) | 4.15 (4.10 to 4.21) | 6.37 (6.22 to 6.46) | 7.86 (7.78 to 7.89) |
| 512, 720 | 277.65 (277.64 to 277.74) | OOM | 7.44 (7.32 to 7.48) | 10.46 (10.10 to 10.48) | 13.79 (13.75 to 13.82) |
| peak MiB at 512, 360 | n/a | 9020 | 24 | 76 | 76 |

**table2**

| Scans | Method | Median error (bins) | Worst error (bins) | Time (s) |
|---|---|---|---|---|
| HTC 2022 (5) | gradient (ours) | 0.004 | 0.013 | 0.22 |
| HTC 2022 (5) | grid | 0.000 | 0.020 | 0.56 |
| HTC 2022 (5) | Vo's method | 0.100 | 0.200 | 0.56 |
| HTC 2022 (5) | phase correlation | 0.095 | 44.149 | 0.00 |
| walnut, 1200 views | gradient (ours) | 0.007 | 0.014 | 0.31 |
| walnut, 1200 views | grid | 0.010 | 0.020 | 0.85 |
| walnut, 1200 views | Vo's method | 0.125 | 0.200 | 0.81 |
| walnut, 1200 views | phase correlation | 31.731 | 84.727 | 0.00 |
| walnut, 120 views | gradient (ours) | 0.024 | 0.031 | 0.12 |
| walnut, 120 views | grid | 0.040 | 0.040 | 0.41 |
| walnut, 120 views | Vo's method | 0.125 | 0.200 | 0.14 |
| walnut, 120 views | phase correlation | 82.138 | 212.929 | 0.00 |

**supplement_gradient_accuracy**

| Operator | Parallel, CUDA f32 | Parallel, PyTorch f32 | Fan, CUDA f32 | Fan, PyTorch f32 |
|---|---|---|---|---|
| forward | 1.4e-02 | 1.2e-02 | 1.1e-02 | 2.2e-02 |
| adjoint | 5.8e-03 | 5.3e-03 | 4.4e-03 | 1.5e-02 |
| backproject | 1.5e-02 | 8.2e-03 | 1.3e-02 | 1.3e-02 |
| fbp | 4.6e-03 | 3.6e-03 | 5.3e-03 | 6.7e-03 |
