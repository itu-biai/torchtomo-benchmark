| Case | Library | Forward rel. L2 | Adjoint rel. L2 | Dot test |
| --- | --- | ---: | ---: | ---: |
| cone 32^3, 90 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 32^3, 90 views | torchtomo-pytorch | 0.0000 | 0.0000 | 1.0000 |
| cone 64^3, 180 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 64^3, 180 views | torchtomo-pytorch | 0.0000 | 0.0000 | 1.0000 |
| cone 128^3, 360 views | torchtomo | 0.0000 | 0.0000 | 1.0000 |
| cone 128^3, 360 views | torchtomo-pytorch | 0.0000 | 0.0000 | 1.0000 |

| Case | Library | Operation | ms | PyTorch peak MB |
| --- | --- | --- | ---: | ---: |
| cone 32^3, 90 views | torchtomo | forward | 0.4 | 1 |
| cone 32^3, 90 views | torchtomo | adjoint | 0.4 | 1 |
| cone 32^3, 90 views | torchtomo | forward + geometry gradient | 0.9 | 6 |
| cone 32^3, 90 views | torchtomo-pytorch | forward | 101.1 | 1699 |
| cone 32^3, 90 views | torchtomo-pytorch | adjoint | 105.6 | 1698 |
| cone 32^3, 90 views | torchtomo-pytorch | forward + geometry gradient | 3965.7 | 4135 |
| cone 64^3, 180 views | torchtomo | forward | 2.3 | 10 |
| cone 64^3, 180 views | torchtomo | adjoint | 2.7 | 4 |
| cone 64^3, 180 views | torchtomo | forward + geometry gradient | 4.3 | 50 |
| cone 64^3, 180 views | torchtomo-pytorch | forward | 1612.2 | 1703 |
| cone 64^3, 180 views | torchtomo-pytorch | adjoint | 1769.4 | 1697 |
| cone 64^3, 180 views | torchtomo-pytorch | forward + geometry gradient | out of memory |  |
| cone 128^3, 360 views | torchtomo | forward | 32.3 | 72 |
| cone 128^3, 360 views | torchtomo | adjoint | 37.6 | 27 |
| cone 128^3, 360 views | torchtomo | forward + geometry gradient | 50.5 | 390 |
| cone 128^3, 360 views | torchtomo-pytorch | forward | 25788.9 | 1748 |
| cone 128^3, 360 views | torchtomo-pytorch | adjoint | 27934.7 | 1704 |
| cone 128^3, 360 views | torchtomo-pytorch | forward + geometry gradient | out of memory |  |
