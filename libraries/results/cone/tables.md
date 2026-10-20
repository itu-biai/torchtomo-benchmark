| Case | Library | Forward rel. L2 | Adjoint rel. L2 | Dot test |
| --- | --- | ---: | ---: | ---: |
| cone 32^3, 90 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 32^3, 90 views | astra | 0.0194 | 0.0272 | 1.0054 |
| cone 32^3, 90 views | leap | 0.0204 | 0.0276 | 1.0628 |
| cone 64^3, 180 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 64^3, 180 views | astra | 0.0107 | 0.0195 | 1.0054 |
| cone 64^3, 180 views | leap | 0.0074 | 0.0196 | 1.0319 |
| cone 128^3, 360 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 128^3, 360 views | astra | 0.0051 | 0.0140 | 1.0053 |
| cone 128^3, 360 views | leap | 0.0041 | 0.0140 | 1.0159 |
| cone 256^3, 360 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 256^3, 360 views | astra | 0.0025 | 0.0102 | 1.0053 |
| cone 256^3, 360 views | leap | 0.0022 | 0.0100 | 1.0079 |

| Case | Library | Operation | ms | PyTorch peak MB |
| --- | --- | --- | ---: | ---: |
| cone 32^3, 90 views | torchtomo | forward | 0.4 | 1 |
| cone 32^3, 90 views | torchtomo | adjoint | 0.5 | 1 |
| cone 32^3, 90 views | torchtomo | forward + geometry gradient | 1.0 | 6 |
| cone 32^3, 90 views | astra | forward | 1.6 | 2 |
| cone 32^3, 90 views | astra | adjoint | 1.1 | 1 |
| cone 32^3, 90 views | leap | forward | 0.4 | 2 |
| cone 32^3, 90 views | leap | adjoint | 0.6 | 1 |
| cone 64^3, 180 views | torchtomo | forward | 2.2 | 10 |
| cone 64^3, 180 views | torchtomo | adjoint | 2.7 | 4 |
| cone 64^3, 180 views | torchtomo | forward + geometry gradient | 4.2 | 50 |
| cone 64^3, 180 views | astra | forward | 2.5 | 13 |
| cone 64^3, 180 views | astra | adjoint | 1.4 | 8 |
| cone 64^3, 180 views | leap | forward | 1.9 | 13 |
| cone 64^3, 180 views | leap | adjoint | 1.2 | 8 |
| cone 128^3, 360 views | torchtomo | forward | 32.1 | 72 |
| cone 128^3, 360 views | torchtomo | adjoint | 37.4 | 27 |
| cone 128^3, 360 views | torchtomo | forward + geometry gradient | 50.1 | 390 |
| cone 128^3, 360 views | astra | forward | 13.9 | 106 |
| cone 128^3, 360 views | astra | adjoint | 5.2 | 61 |
| cone 128^3, 360 views | leap | forward | 25.7 | 107 |
| cone 128^3, 360 views | leap | adjoint | 11.2 | 61 |
| cone 256^3, 360 views | torchtomo | forward | 262.1 | 356 |
| cone 256^3, 360 views | torchtomo | adjoint | 332.2 | 208 |
| cone 256^3, 360 views | torchtomo | forward + geometry gradient | 397.0 | 1630 |
| cone 256^3, 360 views | astra | forward | 101.5 | 425 |
| cone 256^3, 360 views | astra | adjoint | 31.7 | 279 |
| cone 256^3, 360 views | leap | forward | 207.2 | 426 |
| cone 256^3, 360 views | leap | adjoint | 82.6 | 280 |

| Case | Library | Method | PSNR dB | Seconds |
| --- | --- | --- | ---: | ---: |
| cone 32^3, 90 views | torchtomo | SIRT, 30 iterations | 18.79 | 0.0 |
| cone 32^3, 90 views | astra | SIRT, 30 iterations | 18.95 | 0.1 |
| cone 32^3, 90 views | astra | FDK | 18.52 | 0.0 |
| cone 32^3, 90 views | leap | SIRT, 30 iterations | 18.90 | 0.0 |
| cone 64^3, 180 views | torchtomo | SIRT, 30 iterations | 19.27 | 0.2 |
| cone 64^3, 180 views | astra | SIRT, 30 iterations | 19.37 | 0.1 |
| cone 64^3, 180 views | astra | FDK | 21.77 | 0.0 |
| cone 64^3, 180 views | leap | SIRT, 30 iterations | 19.34 | 0.1 |
| cone 128^3, 360 views | torchtomo | SIRT, 30 iterations | 20.10 | 3.3 |
| cone 128^3, 360 views | astra | SIRT, 30 iterations | 20.15 | 0.6 |
| cone 128^3, 360 views | astra | FDK | 24.94 | 0.0 |
| cone 128^3, 360 views | leap | SIRT, 30 iterations | 20.13 | 1.2 |
| cone 256^3, 360 views | torchtomo | SIRT, 30 iterations | 20.60 | 29.2 |
| cone 256^3, 360 views | astra | SIRT, 30 iterations | 20.62 | 4.2 |
| cone 256^3, 360 views | astra | FDK | 27.82 | 0.1 |
| cone 256^3, 360 views | leap | SIRT, 30 iterations | 20.61 | 9.2 |
