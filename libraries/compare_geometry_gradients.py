"""Cost of a geometry gradient: torchtomo against Thies et al.'s backprojector.

Run from the repository root, with geometry_gradients_CT cloned somewhere:
    python libraries/compare_geometry_gradients.py --thies ~/lab/ext/geometry_gradients_CT

Thies et al. (PMB 2023, github.com/mareikethies/geometry_gradients_CT) give a
fan-beam backprojector whose analytic derivative with respect to each view's
projection matrix runs in numba CUDA. It has no forward projector. torchtomo
differentiates forward, adjoint and FBP; the first two run on its CUDA kernels,
the FBP backprojection on the PyTorch path.

Every case is one backprojection (or projection) of a batch of one plus the
gradient of a scalar loss with respect to a per-view lateral translation, the
motion both libraries were built for. Times are the median of repeated runs
after a warm-up, peak memory is what PyTorch allocated. The table is also
written to libraries/results/geometry-gradients.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from torchtomo import FanBeam

RESULTS = Path(__file__).resolve().parent / "results"


def timed(step, repeats):
    step()
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times)), torch.cuda.max_memory_allocated() / 2**20


def thies_case(size, views, n_det, repeats):
    from backprojector_fan import DifferentiableFanBeamBackprojector
    from helper import params_2_proj_matrix

    from geometry import Geometry

    device = torch.device("cuda")
    spacing = 1.0
    geometry = Geometry(
        (size, size), (-(size - 1) / 2, -(size - 1) / 2), (1.0, 1.0), -(n_det - 1) / 2 * spacing, spacing
    )
    angles = np.linspace(0, 2 * np.pi, views, endpoint=False)
    zeros = np.zeros_like(angles)
    matrices, _, _ = params_2_proj_matrix(
        angles,
        4.0 * size * np.ones_like(angles),
        2.0 * size * np.ones_like(angles),
        zeros,
        zeros,
        spacing,
        (n_det - 1) / 2,
    )
    matrices = torch.from_numpy(matrices.astype(np.float32)).to(device)
    sinogram = torch.rand(views, n_det, device=device)
    translation = torch.zeros(views, 2, 1, device=device, requires_grad=True)
    rotation = torch.eye(2, device=device).expand(views, 2, 2)
    bottom = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(views, 1, 3)
    backprojector = DifferentiableFanBeamBackprojector.apply

    def step():
        motion = torch.cat((torch.cat((rotation, translation), dim=2), bottom), dim=1)
        image = backprojector(sinogram, torch.einsum("nij,njk->nik", matrices, motion), geometry)
        image.pow(2).mean().backward()

    return timed(step, repeats)


def torchtomo_case(size, views, n_det, operator, repeats, backend="auto"):
    device = torch.device("cuda")
    projector = FanBeam(img_size=size, n_angles=views, n_det=n_det, backend=backend).to(device)
    base = projector.pose.detach().clone()
    shift = torch.zeros(views, dtype=base.dtype, device=device, requires_grad=True)
    image = torch.rand(1, 1, size, size, device=device)
    sinogram = torch.rand(1, 1, views, n_det, device=device)

    def step():
        projector.pose = torch.stack([base[:, 0], base[:, 1] + shift, base[:, 2] + shift], dim=1)
        if operator == "forward":
            output = projector.forward(image)
        elif operator == "adjoint":
            output = projector.adjoint(sinogram)
        else:
            output = projector.backproject(sinogram)
        output.pow(2).mean().backward()

    return timed(step, repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thies", required=True, help="path to a clone of geometry_gradients_CT")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backends", nargs="+", default=["auto"], choices=["auto", "torch"], help="torchtomo backends")
    parser.add_argument("--output", type=Path, default=RESULTS / "geometry-gradients.json")
    args = parser.parse_args()
    sys.path.insert(0, args.thies)
    # numba-cuda 0.30 still registers np.row_stack, which NumPy 2.4 removed.
    if not hasattr(np, "row_stack"):
        np.row_stack = np.vstack

    cases = [(256, 360, 384), (512, 360, 768), (512, 720, 768)]
    records = []
    print(f"{'size':>5} {'views':>5} {'bins':>5}  {'method':38s} {'ms':>9} {'peak MiB':>9}")
    for size, views, n_det in cases:
        rows = [("Thies et al. backprojection", lambda: thies_case(size, views, n_det, args.repeats))]
        for backend in args.backends:
            prefix = "torchtomo" if backend == "auto" else f"torchtomo {backend}"
            for operator in ("forward", "adjoint", "backproject"):
                rows.append(
                    (
                        f"{prefix} {operator}",
                        lambda operator=operator, backend=backend: torchtomo_case(
                            size, views, n_det, operator, args.repeats, backend
                        ),
                    )
                )
        for label, run in rows:
            torch.cuda.empty_cache()
            try:
                milliseconds, peak = run()
                print(f"{size:5d} {views:5d} {n_det:5d}  {label:38s} {milliseconds:9.2f} {peak:9.0f}", flush=True)
            except torch.OutOfMemoryError:
                milliseconds, peak = None, None
                print(f"{size:5d} {views:5d} {n_det:5d}  {label:38s} {'OOM':>9}", flush=True)
            records.append(
                dict(size=size, views=views, bins=n_det, method=label, milliseconds=milliseconds, peak_mib=peak)
            )
    device = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(device=device, repeats=args.repeats, cases=records), indent=1))


if __name__ == "__main__":
    main()
