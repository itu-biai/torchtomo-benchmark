| Case | Library | Forward rel. L2 | Adjoint rel. L2 | Dot test |
| --- | --- | ---: | ---: | ---: |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 1.36 mm bins | astra | 0.0026 | 0.0402 | 1.0444 |
| VICTRE 694607065 (scattered), 1.36 mm bins | leap | 0.0107 | 0.1590 | 1.0308 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 0.68 mm bins | astra | 0.0046 | 0.0427 | 1.0447 |
| VICTRE 694607065 (scattered), 0.68 mm bins | leap | 0.0023 | 0.0767 | 1.0378 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 0.34 mm bins | astra | 0.0057 | 0.0423 | 1.0448 |
| VICTRE 694607065 (scattered), 0.34 mm bins | leap | 0.0031 | 0.0469 | 1.0413 |

| Case | Library | Operation | ms | PyTorch peak MB |
| --- | --- | --- | ---: | ---: |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | forward | 0.5 | 6 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | adjoint | 0.7 | 7 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | forward + geometry gradient | 1.4 | 17 |
| VICTRE 694607065 (scattered), 1.36 mm bins | astra | forward | 1.3 | 4 |
| VICTRE 694607065 (scattered), 1.36 mm bins | astra | adjoint | 1.3 | 4 |
| VICTRE 694607065 (scattered), 1.36 mm bins | leap | forward | 0.6 | 6 |
| VICTRE 694607065 (scattered), 1.36 mm bins | leap | adjoint | 1.2 | 4 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | forward | 1.6 | 24 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | adjoint | 2.8 | 25 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | forward + geometry gradient | 3.8 | 68 |
| VICTRE 694607065 (scattered), 0.68 mm bins | astra | forward | 1.9 | 14 |
| VICTRE 694607065 (scattered), 0.68 mm bins | astra | adjoint | 1.5 | 16 |
| VICTRE 694607065 (scattered), 0.68 mm bins | leap | forward | 1.7 | 26 |
| VICTRE 694607065 (scattered), 0.68 mm bins | leap | adjoint | 6.5 | 19 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | forward | 9.2 | 97 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | adjoint | 15.5 | 99 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | forward + geometry gradient | 20.8 | 269 |
| VICTRE 694607065 (scattered), 0.34 mm bins | astra | forward | 9.9 | 59 |
| VICTRE 694607065 (scattered), 0.34 mm bins | astra | adjoint | 3.2 | 63 |
| VICTRE 694607065 (scattered), 0.34 mm bins | leap | forward | 8.7 | 151 |
| VICTRE 694607065 (scattered), 0.34 mm bins | leap | adjoint | 49.6 | 124 |

| Case | Library | FBP corr. | OS-SART corr. | Held-out residual | OS-SART seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | 0.857 | 0.431 | 3.54% | 0.0 |
| VICTRE 694607065 (scattered), 1.36 mm bins | astra | 0.865 | 0.433 | 3.53% | 0.1 |
| VICTRE 694607065 (scattered), 1.36 mm bins | leap | 0.793 | 0.423 | 3.60% | 0.1 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | 0.884 | 0.449 | 3.57% | 0.1 |
| VICTRE 694607065 (scattered), 0.68 mm bins | astra | 0.892 | 0.455 | 3.59% | 0.1 |
| VICTRE 694607065 (scattered), 0.68 mm bins | leap | 0.879 | 0.454 | 3.62% | 0.1 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | 0.884 | 0.435 | 3.65% | 0.3 |
| VICTRE 694607065 (scattered), 0.34 mm bins | astra | 0.886 | 0.438 | 3.66% | 0.3 |
| VICTRE 694607065 (scattered), 0.34 mm bins | leap | 0.888 | 0.438 | 3.68% | 0.6 |
