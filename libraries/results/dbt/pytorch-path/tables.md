| Case | Library | Forward rel. L2 | Adjoint rel. L2 | Dot test |
| --- | --- | ---: | ---: | ---: |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo-pytorch | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo-pytorch | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo-pytorch | 0.0000 | 0.0000 | 1.0000 |

| Case | Library | Operation | ms | PyTorch peak MB |
| --- | --- | --- | ---: | ---: |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | forward | 0.5 | 6 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | adjoint | 0.7 | 7 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo | forward + geometry gradient | 1.4 | 17 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo-pytorch | forward | 188.8 | 1700 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo-pytorch | adjoint | 206.0 | 1701 |
| VICTRE 694607065 (scattered), 1.36 mm bins | torchtomo-pytorch | forward + geometry gradient | 7484.6 | 6448 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | forward | 1.6 | 24 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | adjoint | 2.8 | 25 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo | forward + geometry gradient | 3.9 | 68 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo-pytorch | forward | 1494.4 | 1704 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo-pytorch | adjoint | 1629.3 | 1705 |
| VICTRE 694607065 (scattered), 0.68 mm bins | torchtomo-pytorch | forward + geometry gradient | out of memory |  |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | forward | 9.4 | 97 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | adjoint | 15.7 | 99 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo | forward + geometry gradient | 21.1 | 269 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo-pytorch | forward | 11985.3 | 1725 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo-pytorch | adjoint | 13175.0 | 1727 |
| VICTRE 694607065 (scattered), 0.34 mm bins | torchtomo-pytorch | forward + geometry gradient | out of memory |  |
