"""Relative error of the float32 CUDA geometry gradient against the float64 PyTorch path.

Run from the repository root:
    python isbi/gradient_accuracy.py

For each geometry and operator (forward, adjoint, backproject, fbp), the loss
0.5 ||O_p(z)||^2 of a fixed random input z is differentiated with respect to the
whole pose table p (angles and shifts), on torchtomo's CUDA kernels in float32,
on the PyTorch path in float32 (the roundoff floor) and on the PyTorch path in
float64 (the reference). Writes isbi/results/gradient_accuracy.json.

The pose is the nominal one plus a seeded random offset per view (angles within
+-0.5 degrees, shifts within +-0.5 px). The pose gradient runs through linear
interpolation, whose derivative jumps where a sample lands exactly on a pixel or
bin centre; the nominal parallel geometry (angles 0 and 90 degrees, no shift)
sits on those kinks, where float32 and float64 rounding pick different one-sided
derivatives. A generic pose measures the gradient, not the kink.
"""

import argparse
import json
from pathlib import Path

import torch
from provenance import current, stamp

from torchtomo import FanBeam, ParallelBeam

RESULTS = Path(__file__).resolve().parent / "results"
OPERATORS: tuple[str, ...] = ("forward", "adjoint", "backproject", "fbp")
GEOMETRIES = {"parallel": ParallelBeam, "fan": FanBeam}


def pose_gradient(projector: torch.nn.Module, operator: str, data: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    leaf = pose.to(device=data.device, dtype=data.dtype).clone().requires_grad_(True)
    projector.pose = leaf
    output = getattr(projector, operator)(data)
    (gradient,) = torch.autograd.grad(0.5 * output.double().pow(2).sum(), leaf)
    return gradient.detach().double().cpu()


def relative_errors(geometry: str, operator: str, size: int, views: int, seed: int) -> dict[str, float]:
    device = torch.device("cuda")
    build = GEOMETRIES[geometry]
    fast = build(img_size=size, n_angles=views, backend="cuda").to(device)
    floor = build(img_size=size, n_angles=views, backend="torch").to(device)
    exact = build(img_size=size, n_angles=views, backend="torch").double().to(device)
    generator = torch.Generator().manual_seed(seed)
    base = exact.pose.detach().cpu()
    offset = torch.rand(base.shape, generator=generator, dtype=torch.float64) - 0.5
    offset[:, 0] *= torch.pi / 180  # +-0.5 degrees
    pose = base + offset  # shifts +-0.5 px
    shape = (1, 1, size, size) if operator == "forward" else (1, 1, views, fast.n_det)
    data = torch.rand(shape, generator=generator, dtype=torch.float64)
    reference = pose_gradient(exact, operator, data.to(device), pose)
    gradients = {
        "cuda32": pose_gradient(fast, operator, data.float().to(device), pose),
        "torch32": pose_gradient(floor, operator, data.float().to(device), pose),
    }
    out = {}
    for name, gradient in gradients.items():
        out[f"{name}_relative_error"] = float((gradient - reference).norm() / reference.norm())
        for column in range(reference.shape[1]):
            out[f"{name}_column_{column}"] = float(
                (gradient[:, column] - reference[:, column]).norm() / reference[:, column].norm()
            )
    out["relative_error"] = out["cuda32_relative_error"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--views", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    rows = []
    for geometry in GEOMETRIES:
        for operator in OPERATORS:
            for seed in args.seeds:
                errors = relative_errors(geometry, operator, args.size, args.views, seed)
                rows.append({"geometry": geometry, "operator": operator, "seed": seed, **errors})
                print(
                    f"{geometry:8s} {operator:11s} seed {seed}  cuda32 {errors['cuda32_relative_error']:.3e}  "
                    f"torch32 {errors['torch32_relative_error']:.3e}",
                    flush=True,
                )
    raw = RESULTS / "raw" / "gradient_accuracy.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"size": args.size, "views": args.views, "rows": rows}, indent=1))
    stamp(raw, RESULTS / "gradient_accuracy.json", current("python isbi/gradient_accuracy.py"))


if __name__ == "__main__":
    main()
