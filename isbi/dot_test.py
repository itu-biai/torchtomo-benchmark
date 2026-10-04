"""Adjoint exactness of every available operator pair: the dot test.

Run from the repository root (needs CUDA; ASTRA and torch-radon rows appear
when those are installed):
    python isbi/dot_test.py

For random standard-normal pairs (x, y) the normalized defect is
|<Ax, y> - <x, A^T y>| / (||Ax|| ||y|| + ||x|| ||A^T y||), where A^T is what
each library offers as the adjoint (torchtomo's adjoint(), ASTRA's and
torch-radon's backprojection). Inner products are accumulated in float64.
Writes isbi/results/dot_test.json.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from provenance import current, stamp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libraries"))
RESULTS = Path(__file__).resolve().parent / "results"

from torchtomo import FanBeam, ParallelBeam  # noqa: E402

Pair = tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]


def torchtomo_pair(geometry: str, size: int, views: int, **options: object) -> Pair:
    build = ParallelBeam if geometry == "parallel" else FanBeam
    projector = build(img_size=size, n_angles=views, **options).cuda()
    return projector.forward, projector.adjoint


def pairs(geometry: str, size: int, views: int) -> dict[str, Callable[[], Pair]]:
    rows: dict[str, Callable[[], Pair]] = {
        "torchtomo CUDA exact": lambda: torchtomo_pair(geometry, size, views, backend="cuda"),
        "torchtomo CUDA approximate": lambda: torchtomo_pair(geometry, size, views, backend="cuda", approximate=True),
        "torchtomo PyTorch": lambda: torchtomo_pair(geometry, size, views, backend="torch"),
    }
    try:
        from astra_projector import AstraFanBeam, AstraParallelBeam

        astra_class = AstraParallelBeam if geometry == "parallel" else AstraFanBeam

        def astra_pair() -> Pair:
            projector = astra_class(img_size=size, n_angles=views).cuda()
            return projector.forward, projector.adjoint

        rows["ASTRA"] = astra_pair
    except ImportError as error:
        print(f"ASTRA unavailable: {error}")
    try:
        from speed_table import TorchRadonOperators

        def radon_pair() -> Pair:
            operators = TorchRadonOperators(geometry, size, views)
            return operators.forward, operators.adjoint

        rows["torch-radon"] = radon_pair
    except ImportError as error:
        print(f"torch-radon unavailable: {error}")
    return rows


def defects(pair: Pair, size: int, count: int, seed: int) -> torch.Tensor:
    forward, adjoint = pair
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(count):
        x = torch.randn(1, 1, size, size, generator=generator).cuda()
        with torch.no_grad():
            ax = forward(x)
            y = torch.randn(ax.shape, generator=generator).cuda()
            aty = adjoint(y)
        lhs = (ax.double() * y.double()).sum()
        rhs = (x.double() * aty.double()).sum()
        scale = ax.double().norm() * y.double().norm() + x.double().norm() * aty.double().norm()
        values.append(float((lhs - rhs).abs() / scale))
    return torch.tensor(values, dtype=torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for geometry in ("parallel", "fan"):
        for name, build in pairs(geometry, args.size, args.views).items():
            try:
                values = defects(build(), args.size, args.pairs, args.seed)
            except Exception as error:  # an optional library failing must not hide the others
                print(f"{geometry:8s} {name:28s} failed: {error}")
                rows.append({"geometry": geometry, "pair": name, "error": str(error)})
                continue
            row = {
                "geometry": geometry,
                "pair": name,
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
            rows.append(row)
            print(f"{geometry:8s} {name:28s} median {row['median']:.2e}  p95 {row['p95']:.2e}", flush=True)
    raw = RESULTS / "raw" / "dot_test.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"size": args.size, "views": args.views, "pairs": args.pairs, "rows": rows}, indent=1))
    stamp(raw, RESULTS / "dot_test.json", current(f"python isbi/dot_test.py --pairs {args.pairs}"))


if __name__ == "__main__":
    main()
