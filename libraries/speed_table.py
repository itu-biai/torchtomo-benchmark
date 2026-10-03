#!/usr/bin/env python
"""torchtomo's backends, LEAP, torch-radon, ASTRA and TIGRE timed in one process.

Forward, adjoint, and FBP in milliseconds per call on the same batch. Running
every projector in one process on a card that was just spun up keeps clock
ramping and other jobs from favouring whichever ran first. torchtomo's own
columns also come from torchtomo's benchmark/benchmark_speed.py.

    python libraries/speed_table.py --json libraries/results/speed-table.json

Held memory is reported for torchtomo only: the other libraries allocate
outside PyTorch's caching allocator, where torch.cuda.memory_allocated cannot
see them.

ASTRA works on the torch tensors in place through DLPack. TIGRE takes NumPy
arrays, so its rows include the copies to the card and back that every TIGRE
call makes.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from compare_libraries import (
    AstraFanBeam,
    AstraParallelBeam,
    LeapFanBeam,
    LeapParallelBeam,
    Radon,
    RadonFanbeam,
    TigreFanBeam,
    TigreParallelBeam,
    TorchRadonBackend,
    TorchRadonFanBackend,
    angle_tensor,
)

from torchtomo import FanBeam, ParallelBeam
from torchtomo._cuda_kernels import cuda_kernels_available
from torchtomo._triton_kernels import triton_kernels_available


def warm(seconds=2.0):
    """Spin the card so its clocks are up before the first timed row."""
    a = torch.rand(2048, 2048, device="cuda")
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ a
    torch.cuda.synchronize()


def milliseconds(fn, warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


class TorchRadonOperators:
    """torch-radon behind the forward / adjoint / fbp names the rest of this script calls."""

    def __init__(self, geometry, size, n_angles):
        device = torch.device("cuda")
        backend_class = TorchRadonBackend if geometry == "parallel" else TorchRadonFanBackend
        self.backend = backend_class(size, angle_tensor(n_angles, device, geometry), device)
        self.n_det = self.backend.forward(torch.zeros(1, 1, size, size, device=device)).shape[-1]

    def forward(self, image):
        return self.backend.forward(image)

    def adjoint(self, sinogram):
        return self.backend.backproject(sinogram)

    def fbp(self, sinogram):
        return self.backend.fbp(sinogram)


def projectors(geometry, size, n_angles):
    """(name, is torchtomo, constructor) for every projector that can run here."""
    cuda = torch.device("cuda")
    torchtomo_class = ParallelBeam if geometry == "parallel" else FanBeam
    rows = [("torch", True, lambda: torchtomo_class(img_size=size, n_angles=n_angles, backend="torch").cuda())]
    if geometry == "parallel" and triton_kernels_available(cuda, torch.float32):
        rows.append(("triton", True, lambda: ParallelBeam(img_size=size, n_angles=n_angles, backend="triton").cuda()))
    if cuda_kernels_available(cuda, torch.float32):
        rows.append(("cuda", True, lambda: torchtomo_class(img_size=size, n_angles=n_angles, backend="cuda").cuda()))
        rows.append(
            (
                "cuda, approximate",
                True,
                lambda: torchtomo_class(img_size=size, n_angles=n_angles, backend="cuda", approximate=True).cuda(),
            )
        )
    leap_class = LeapParallelBeam if geometry == "parallel" else LeapFanBeam
    if leap_class is not None:
        rows.append(("LEAP", False, lambda: leap_class(img_size=size, n_angles=n_angles).cuda()))
    if (Radon if geometry == "parallel" else RadonFanbeam) is not None:
        rows.append(("torch-radon", False, lambda: TorchRadonOperators(geometry, size, n_angles)))
    for name, adapter in (("ASTRA", (AstraParallelBeam, AstraFanBeam)), ("TIGRE", (TigreParallelBeam, TigreFanBeam))):
        adapter_class = adapter[0] if geometry == "parallel" else adapter[1]
        if adapter_class is not None:
            rows.append(
                (
                    name,
                    False,
                    lambda adapter_class=adapter_class: adapter_class(img_size=size, n_angles=n_angles).cuda(),
                )
            )
    return rows


@torch.no_grad()
def time_operators(projector, image, sinogram):
    return dict(
        forward_ms=milliseconds(lambda: projector.forward(image)),
        adjoint_ms=milliseconds(lambda: projector.adjoint(sinogram)),
        fbp_ms=milliseconds(lambda: projector.fbp(sinogram)),
    )


def run(args):
    rows = []
    warm()
    for geometry in args.geometries:
        for size in args.sizes:
            for n_angles in args.angles:
                image = torch.rand(args.batch_size, 1, size, size, device="cuda")
                for name, is_torchtomo, build in projectors(geometry, size, n_angles):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_allocated()
                    projector = build()
                    sinogram = torch.rand(args.batch_size, 1, n_angles, projector.n_det, device="cuda")
                    row = dict(
                        geometry=geometry,
                        size=size,
                        angles=n_angles,
                        batch=args.batch_size,
                        projector=name,
                        **time_operators(projector, image, sinogram),
                    )
                    torch.cuda.synchronize()
                    held = torch.cuda.memory_allocated() - before - sinogram.nbytes
                    row["held_mb"] = held / 2**20 if is_torchtomo else None
                    rows.append(row)
                    print(format_row(row), flush=True)
                    del projector, sinogram
    return rows


def format_row(row):
    held = "" if row["held_mb"] is None else f"{row['held_mb']:.0f}"
    return (
        f"| {row['geometry']} | {row['size']} | {row['angles']} | {row['projector']} "
        f"| {row['forward_ms']:.2f} | {row['adjoint_ms']:.2f} | {row['fbp_ms']:.2f} | {held} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometries", nargs="+", choices=("parallel", "fan"), default=["parallel", "fan"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[512])
    parser.add_argument("--angles", nargs="+", type=int, default=[360, 90])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this comparison needs CUDA: the other libraries are CUDA only")

    print(f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, batch {args.batch_size}")
    print("| geometry | size | angles | projector | forward ms | adjoint ms | FBP ms | held MB |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = run(args)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
