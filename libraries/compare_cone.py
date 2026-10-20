"""torchtomo's 3D cone beam and breast tomosynthesis against ASTRA and LEAP.

The three libraries get one geometry: torchtomo's source positions and detector
element centres, handed to ASTRA as ``cone_vec`` and to LEAP as a modular beam
by ``cone_projectors.py``. Two scans are measured:

- ``cone``: a circular flat-panel scan of a 3D Shepp-Logan phantom, magnification
  2, over a full turn.
- ``dbt``: the published VICTRE arc (25 views over 50 degrees, stationary
  detector, source over the chest wall) on the measured projections and VICTRE's
  own reconstruction of one TCIA phantom, prepared by torchtomo's
  ``benchmark/dbt/prepare_victre.py``.

Sections:

- ``agreement``: forward projections and backprojections against torchtomo's,
  relative L2 with no fitted scale, and each library's dot-product test.
- ``performance``: milliseconds per forward and backprojection of one volume, and
  the PyTorch allocation peak. torchtomo also times a forward with a geometry
  gradient, which the others do not offer.
- ``quality``: every library's operators in the same reconstruction code. Cone:
  SIRT on its own noiseless projections, scored by PSNR against the phantom, and
  ASTRA's FDK. DBT: the dataset's FBP filter followed by each backprojector, and
  OS-SART fitted to the even views; scored by correlation with VICTRE's FBP in
  the breast and by the residual on the held-out odd views.

    python libraries/compare_cone.py --geometry cone
    python libraries/compare_cone.py --geometry dbt \\
        --victre ../torchtomo/benchmark/dbt/results/victre/input_bin4.npz

``torchtomo`` is ConeBeam's CUDA kernels (``backend="cuda"``), the reference the
agreement is measured against; ``--libraries torchtomo-pytorch`` adds its PyTorch
path. ConeBeam, BreastTomosynthesis, and the kernels are newer than torchtomo
0.4.0, so these need a torchtomo checkout that has them on the path.
"""

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from benchmark_metadata import benchmark_metadata

import torchtomo
from torchtomo import ConeBeam

try:
    from cone_projectors import AstraConeBeam
except ImportError:
    AstraConeBeam = None

try:
    from cone_projectors import LeapConeBeam, tomographicModels
except ImportError:
    LeapConeBeam = tomographicModels = None

CONE_CASES = ((32, 90), (64, 180), (128, 360), (256, 360))
DBT_BINS = (16, 8, 4)
SAMPLES_PER_CHUNK = 2**21

# Modified Shepp-Logan head (Kak and Slaney, with Toft's contrasts) on [-1, 1]^3:
# value, semi-axes (x, y, z), centre (x, y, z), rotation about z in degrees.
SHEPP_LOGAN_3D = (
    (1.0, (0.69, 0.92, 0.81), (0.0, 0.0, 0.0), 0),
    (-0.8, (0.6624, 0.874, 0.78), (0.0, -0.0184, 0.0), 0),
    (-0.2, (0.11, 0.31, 0.22), (0.22, 0.0, 0.0), -18),
    (-0.2, (0.16, 0.41, 0.28), (-0.22, 0.0, 0.0), 18),
    (0.1, (0.21, 0.25, 0.41), (0.0, 0.35, -0.15), 0),
    (0.1, (0.046, 0.046, 0.05), (0.0, 0.1, 0.25), 0),
    (0.1, (0.046, 0.046, 0.05), (0.0, -0.1, 0.25), 0),
    (0.1, (0.046, 0.023, 0.05), (-0.08, -0.605, 0.0), 0),
    (0.1, (0.023, 0.023, 0.02), (0.0, -0.606, 0.0), 0),
    (0.1, (0.023, 0.046, 0.02), (0.06, -0.605, 0.0), 0),
)


class Case:
    """One geometry: the volume grid, the scan, and what the quality section needs."""

    def __init__(self, name, volume_shape, sources, detectors, voxel_spacing, volume_center, n_samples, **extra):
        self.name = name
        self.volume_shape = tuple(volume_shape)
        self.sources = sources
        self.detectors = detectors
        self.voxel_spacing = tuple(float(value) for value in voxel_spacing)
        self.volume_center = tuple(float(value) for value in volume_center)
        self.n_samples = int(n_samples)
        self.extra = extra

    def subset(self, views, depth_margin=0):
        """The same scan restricted to some views, with extra slices above and below."""
        depth = self.volume_shape[0] + 2 * depth_margin
        return Case(
            self.name,
            (depth, *self.volume_shape[1:]),
            self.sources[views],
            self.detectors[views],
            self.voxel_spacing,
            self.volume_center,
            math.ceil(self.n_samples * depth / self.volume_shape[0]),
        )

    def describe(self):
        views, rows, columns = self.detectors.shape[:3]
        return {
            "case": self.name,
            "volume_shape_zyx": list(self.volume_shape),
            "voxel_spacing_zyx": list(self.voxel_spacing),
            "volume_center_xyz": list(self.volume_center),
            "views": views,
            "detector_rows_columns": [rows, columns],
            "rays": views * rows * columns,
            "torchtomo_samples_per_ray": self.n_samples,
            "torchtomo_ray_chunk": ray_chunk(self.n_samples),
        }


TORCHTOMO_BACKENDS = {"torchtomo": "cuda", "torchtomo-pytorch": "torch"}


def ray_chunk(n_samples):
    return max(1, SAMPLES_PER_CHUNK // n_samples)


def torchtomo_operator(case, sources, detectors, backend):
    arguments = (case.volume_shape, sources, detectors, case.voxel_spacing, case.volume_center)
    return ConeBeam(*arguments, n_samples=case.n_samples, ray_chunk=ray_chunk(case.n_samples), backend=backend)


def operator(library, case):
    if library in TORCHTOMO_BACKENDS:
        return torchtomo_operator(case, case.sources, case.detectors, TORCHTOMO_BACKENDS[library])
    arguments = (case.volume_shape, case.sources, case.detectors, case.voxel_spacing, case.volume_center)
    return {"astra": AstraConeBeam, "leap": LeapConeBeam}[library](*arguments)


def available_libraries():
    libraries = ["torchtomo"]
    if AstraConeBeam is not None:
        libraries.append("astra")
    if LeapConeBeam is not None and tomographicModels is not None:
        libraries.append("leap")
    return libraries


def circular_scan(size, views, device):
    """Flat panel at magnification 2, source 4 volume widths from the axis, rows along +z."""
    source_to_axis, source_to_detector, pixel = 4.0 * size, 8.0 * size, 2.0
    rows = columns = size * 3 // 2
    angles = torch.arange(views, dtype=torch.float64) * (2 * math.pi / views)
    cosine, sine, zero = angles.cos(), angles.sin(), torch.zeros(views, dtype=torch.float64)
    sources = torch.stack([source_to_axis * cosine, source_to_axis * sine, zero], dim=-1)
    behind = source_to_detector - source_to_axis
    centres = torch.stack([-behind * cosine, -behind * sine, zero], dim=-1)
    along_row = torch.stack([-sine, cosine, zero], dim=-1)
    along_column = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(views, 3)
    row = (torch.arange(rows, dtype=torch.float64) - (rows - 1) / 2) * pixel
    column = (torch.arange(columns, dtype=torch.float64) - (columns - 1) / 2) * pixel
    detectors = (
        centres[:, None, None]
        + row[:, None, None] * along_column[:, None, None]
        + column[:, None] * along_row[:, None, None]
    )
    case = Case(
        f"cone {size}^3, {views} views",
        (size, size, size),
        sources.float().to(device),
        detectors.float().to(device),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        2 * size,
        source_to_axis=source_to_axis,
        source_to_detector=source_to_detector,
        detector_pixel=pixel,
    )
    return case


def shepp_logan_3d(size, device):
    axis = (torch.arange(size, device=device, dtype=torch.float64) - (size - 1) / 2) / (size / 2)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    volume = torch.zeros(size, size, size, device=device, dtype=torch.float64)
    for value, (a, b, c), (x0, y0, z0), degrees in SHEPP_LOGAN_3D:
        cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        u = (x - x0) * cosine + (y - y0) * sine
        v = -(x - x0) * sine + (y - y0) * cosine
        volume += value * ((u / a) ** 2 + (v / b) ** 2 + ((z - z0) / c) ** 2 <= 1)
    return volume.float()[None, None]


def load_victre(path, bin_factor, depth_margin, device):
    """The prepared bin-4 VICTRE scan, pooled to a coarser bin with invalid rays left out."""
    from torchtomo.victre import victre_projector

    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    if metadata["bin_factor"] > bin_factor or bin_factor % metadata["bin_factor"]:
        raise ValueError(f"{path} is binned by {metadata['bin_factor']}, which does not divide {bin_factor}")
    pool = bin_factor // metadata["bin_factor"]
    projections = torch.as_tensor(data["projections"], device=device)[None]
    valid = torch.as_tensor(data["valid"], device=device)[None]
    reference = torch.as_tensor(data["reference"], device=device)[None]
    if pool > 1:
        weight = F.avg_pool2d(valid.float(), pool)
        projections = F.avg_pool2d(projections * valid, pool) / weight.clamp_min(1e-6)
        valid = weight > 0.999
        reference = F.avg_pool2d(reference, pool)
    reference = reference.transpose(-1, -2).contiguous()
    pixel = 0.085 * bin_factor
    depth = reference.shape[1]
    tomosynthesis = victre_projector(metadata["density"], bin_factor=bin_factor, device=device)
    cone = tomosynthesis.operator
    case = Case(
        f"VICTRE {metadata['patient']} ({metadata['density']}), {pixel:.2f} mm bins",
        cone.volume_shape,
        cone.source_positions,
        cone.detector_positions,
        cone.spacing_xyz.flip(0).tolist(),
        cone.volume_center.tolist(),
        math.ceil(1.3 * depth / pixel),
        patient=metadata["patient"],
        density=metadata["density"],
        bin_factor=bin_factor,
        depth_margin=depth_margin,
        projections=projections[None],
        valid=valid[None],
        reference=reference[None],
        tomosynthesis=tomosynthesis,
    )
    return case


def relative_l2(reference, other):
    return float((reference - other).norm() / reference.norm())


def fitted_scale(reference, other):
    return float((reference.double() * other.double()).sum() / reference.double().square().sum())


def dot_test(projector, generator):
    volume = torch.rand(1, 1, *projector.volume_shape, generator=generator, device=generator.device)
    projections = torch.rand(1, 1, *projector.projection_shape, generator=generator, device=generator.device)
    forward = float((projector.forward(volume).double() * projections.double()).sum())
    backward = float((volume.double() * projector.adjoint(projections).double()).sum())
    return forward / backward


def phantom(case, device):
    if "reference" in case.extra:
        return case.extra["reference"].clamp_min(0)
    return shepp_logan_3d(case.volume_shape[0], device)


@torch.no_grad()
def measure_agreement(case, libraries, device):
    volume = phantom(case, device)
    ours = operator("torchtomo", case)
    projections = ours(volume)
    backprojection = ours.adjoint(projections)
    rows = []
    for library in libraries:
        projector = ours if library == "torchtomo" else operator(library, case)
        theirs = projector.forward(volume)
        their_backprojection = projector.adjoint(projections)
        generator = torch.Generator(device=device).manual_seed(2026)
        rows.append(
            {
                **case.describe(),
                "library": library,
                "forward_relative_l2": relative_l2(projections, theirs),
                "forward_scale": fitted_scale(projections, theirs),
                "adjoint_relative_l2": relative_l2(backprojection, their_backprojection),
                "adjoint_scale": fitted_scale(backprojection, their_backprojection),
                "dot_test_ratio": dot_test(projector, generator),
            }
        )
        print(json.dumps({key: rows[-1][key] for key in ("case", "library", "forward_relative_l2", "dot_test_ratio")}))
        del projector
    return rows


def timed(function, min_runs=3, min_seconds=1.0, enough_seconds=60.0):
    """Mean milliseconds per call after one warmup.

    Runs at least min_runs calls and min_seconds, unless the calls so far already
    took enough_seconds, so a call of minutes is timed once.
    """
    function()
    torch.cuda.synchronize()
    runs, started = 0, time.perf_counter()
    while runs < min_runs or time.perf_counter() - started < min_seconds:
        function()
        torch.cuda.synchronize()
        runs += 1
        if time.perf_counter() - started > enough_seconds:
            break
    return 1000 * (time.perf_counter() - started) / runs, runs


def warm(seconds=2.0):
    """Spin the card so its clocks are up before the first timed row."""
    matrix = torch.rand(2048, 2048, device="cuda")
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        matrix @ matrix
    torch.cuda.synchronize()


def geometry_gradient(library, case, volume):
    """A forward projection and the gradient of its squared norm with respect to every position."""
    sources = case.sources.clone().requires_grad_()
    detectors = case.detectors.clone().requires_grad_()
    projector = torchtomo_operator(case, sources, detectors, TORCHTOMO_BACKENDS[library])

    def step():
        projector(volume).square().sum().backward()

    return step


def measure_performance(case, libraries, device):
    volume = phantom(case, device)
    rows = []
    for library in libraries:
        projector = operator(library, case)
        with torch.no_grad():
            projections = projector.forward(volume)
        operations = {"forward": partial(projector.forward, volume), "adjoint": partial(projector.adjoint, projections)}
        if library in TORCHTOMO_BACKENDS:
            operations["forward + geometry gradient"] = geometry_gradient(library, case, volume)
        for name, function in operations.items():
            gc_and_sync()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            row = {**case.describe(), "library": library, "operation": name}
            try:
                with torch.set_grad_enabled(name.endswith("gradient")):
                    row["milliseconds"], row["runs"] = timed(function)
                row["torch_peak_mb"] = (torch.cuda.max_memory_allocated() - before) / 1e6
            except torch.OutOfMemoryError:
                row["milliseconds"] = row["torch_peak_mb"] = None
                row["error"] = "out of memory"
            row["external_allocations_measured"] = library in TORCHTOMO_BACKENDS
            rows.append(row)
            print(json.dumps({key: row.get(key) for key in ("case", "library", "operation", "milliseconds")}))
        del projector, projections, operations
    return rows


def gc_and_sync():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@torch.no_grad()
def sirt(projector, measured, valid, iterations, volume=None):
    """ASTRA's SIRT3D_CUDA update with a sinogram mask and MinConstraint 0."""
    volume = torch.zeros(1, 1, *projector.volume_shape, device=measured.device) if volume is None else volume
    row_sums = projector.forward(torch.ones_like(volume))
    column_sums = projector.adjoint(valid)
    row_weight = torch.where(row_sums > 1e-6, 1 / row_sums, 0) * valid
    column_weight = torch.where(column_sums > 1e-6, 1 / column_sums, 0)
    for _ in range(iterations):
        residual = (measured - projector.forward(volume)) * row_weight
        volume = (volume + projector.adjoint(residual) * column_weight).clamp_min(0)
    return volume


def psnr(truth, image, mask):
    error = float((truth - image)[mask].double().square().mean())
    return 10 * math.log10(1.0 / error)


@torch.no_grad()
def cone_quality(case, libraries, iterations, device):
    truth = phantom(case, device)
    size = case.volume_shape[0]
    axis = torch.arange(size, device=device) - (size - 1) / 2
    inside = (axis[None, :, None] ** 2 + axis[None, None, :] ** 2 <= (size / 2) ** 2).expand(size, size, size)
    mask = inside[None, None]
    rows = []
    for library in libraries:
        projector = operator(library, case)
        measured = projector.forward(truth)
        valid = torch.ones_like(measured)
        torch.cuda.synchronize()
        started = time.perf_counter()
        volume = sirt(projector, measured, valid, iterations)
        torch.cuda.synchronize()
        rows.append(
            {
                **case.describe(),
                "library": library,
                "method": f"SIRT, {iterations} iterations",
                "psnr_db": psnr(truth, volume, mask),
                "seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps({key: rows[-1][key] for key in ("case", "library", "method", "psnr_db", "seconds")}))
        if library == "astra":
            started = time.perf_counter()
            volume = projector.fdk(measured)
            torch.cuda.synchronize()
            rows.append({**rows[-1], "method": "FDK", "psnr_db": psnr(truth, volume, mask)})
            rows[-1]["seconds"] = time.perf_counter() - started
            print(json.dumps({key: rows[-1][key] for key in ("case", "library", "method", "psnr_db")}))
        del projector
    return rows


def correlation(left, right, mask):
    x, y = left[mask].double(), right[mask].double()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def held_out_residual(predicted, measured, valid):
    return float(((predicted - measured) * valid).norm() / (measured * valid).norm())


@torch.no_grad()
def dbt_quality(case, libraries, iterations, subsets, device):
    from torchtomo.victre import N_VIEWS

    breast_mask, hann_ramp_filter = victre_reconstruction_helpers()

    projections, valid, reference = (case.extra[key] for key in ("projections", "valid", "reference"))
    mask = breast_mask(case.extra["tomosynthesis"], projections, valid, reference)
    filtered = hann_ramp_filter(projections * valid, case.extra["tomosynthesis"])
    valid = valid.float()
    training = list(range(0, N_VIEWS, 2))
    held_out = list(range(1, N_VIEWS, 2))
    margin = case.extra["depth_margin"]
    rows = []
    for library in libraries:
        projector = operator(library, case)
        torch.cuda.synchronize()
        started = time.perf_counter()
        fbp = projector.adjoint(filtered)
        torch.cuda.synchronize()
        row = {
            **case.describe(),
            "library": library,
            "mask_voxels": int(mask.sum()),
            "fbp_seconds": time.perf_counter() - started,
            "fbp_reference_correlation": correlation(fbp, reference, mask),
        }
        del projector
        started = time.perf_counter()
        groups = [training[start::subsets] for start in range(subsets)]
        parts = [(operator(library, case.subset(views, margin)), views) for views in groups]
        prepared = []
        volume = torch.zeros(1, 1, *parts[0][0].volume_shape, device=device)
        for projector, views in parts:
            row_sums = projector.forward(torch.ones_like(volume))
            column_sums = projector.adjoint(valid[:, :, views])
            row_weight = torch.where(row_sums > 1e-6, 1 / row_sums, 0) * valid[:, :, views]
            column_weight = torch.where(column_sums > 1e-6, 1 / column_sums, 0)
            prepared.append((projector, projections[:, :, views], row_weight, column_weight))
        for _ in range(iterations):
            for projector, measured, row_weight, column_weight in prepared:
                residual = (measured - projector.forward(volume)) * row_weight
                volume = (volume + projector.adjoint(residual) * column_weight).clamp_min(0)
        torch.cuda.synchronize()
        row["sart_seconds"] = time.perf_counter() - started
        held = operator(library, case.subset(held_out, margin))
        row["sart_held_out_relative_residual"] = held_out_residual(
            held.forward(volume), projections[:, :, held_out], valid[:, :, held_out]
        )
        sart = volume[:, :, margin : margin + reference.shape[2]]
        row["sart_reference_correlation"] = correlation(sart, reference, mask)
        row["sart"] = {"iterations": iterations, "subsets": subsets, "depth_margin_slices": margin}
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if not isinstance(value, (list, dict))}))
        del parts, prepared, held
        gc_and_sync()
    return rows


def victre_reconstruction_helpers():
    """The breast mask and FBP filter of torchtomo's benchmark/dbt/reconstruct_victre.py, next to its source."""

    for parent in Path(torchtomo.__file__).resolve().parents:
        candidate = parent / "benchmark" / "dbt"
        if (candidate / "reconstruct_victre.py").exists():
            sys.path.insert(0, str(candidate))
            from reconstruct_victre import breast_mask, hann_ramp_filter

            return breast_mask, hann_ramp_filter
    raise FileNotFoundError("torchtomo's benchmark/dbt/reconstruct_victre.py is not next to the installed source")


def cases_for(args, device):
    if args.geometry == "cone":
        return [circular_scan(size, views, device) for size, views in args.cone_cases]
    return [load_victre(args.victre, bin_factor, args.depth_margin, device) for bin_factor in args.bins]


def markdown_tables(results):
    lines = ["| Case | Library | Forward rel. L2 | Adjoint rel. L2 | Dot test |", "| --- | --- | ---: | ---: | ---: |"]
    for row in results.get("agreement", []):
        lines.append(
            f"| {row['case']} | {row['library']} | {row['forward_relative_l2']:.4f} | "
            f"{row['adjoint_relative_l2']:.4f} | {row['dot_test_ratio']:.4f} |"
        )
    lines += ["", "| Case | Library | Operation | ms | PyTorch peak MB |", "| --- | --- | --- | ---: | ---: |"]
    for row in results.get("performance", []):
        milliseconds = "out of memory" if row["milliseconds"] is None else f"{row['milliseconds']:.1f}"
        peak = "" if row["torch_peak_mb"] is None else f"{row['torch_peak_mb']:.0f}"
        lines.append(f"| {row['case']} | {row['library']} | {row['operation']} | {milliseconds} | {peak} |")
    quality = results.get("quality", [])
    if quality and "psnr_db" in quality[0]:
        lines += ["", "| Case | Library | Method | PSNR dB | Seconds |", "| --- | --- | --- | ---: | ---: |"]
        for row in quality:
            lines.append(
                f"| {row['case']} | {row['library']} | {row['method']} | {row['psnr_db']:.2f} | {row['seconds']:.1f} |"
            )
    elif quality:
        lines += [
            "",
            "| Case | Library | FBP corr. | OS-SART corr. | Held-out residual | OS-SART seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in quality:
            lines.append(
                f"| {row['case']} | {row['library']} | {row['fbp_reference_correlation']:.3f} | "
                f"{row['sart_reference_correlation']:.3f} | {100 * row['sart_held_out_relative_residual']:.2f}% | "
                f"{row['sart_seconds']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", choices=("cone", "dbt"), default="cone")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("agreement", "performance", "quality"),
        default=["agreement", "performance", "quality"],
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        help="default: every installed one; torchtomo is its CUDA kernels and torchtomo-pytorch its PyTorch path",
    )
    parser.add_argument(
        "--cone-cases",
        type=lambda text: tuple(map(int, text.split("x"))),
        nargs="+",
        default=list(CONE_CASES),
        help="size x views, such as 128x360",
    )
    parser.add_argument("--bins", type=int, nargs="+", default=list(DBT_BINS))
    parser.add_argument("--victre", type=Path, default=Path("../torchtomo/benchmark/dbt/results/victre/input_bin4.npz"))
    parser.add_argument("--sirt-iterations", type=int, default=30)
    parser.add_argument("--sart-iterations", type=int, default=10)
    parser.add_argument("--subsets", type=int, default=5)
    parser.add_argument("--depth-margin", type=int, default=8)
    parser.add_argument("--quality-max-size", type=int, default=256, help="largest cone volume reconstructed")
    parser.add_argument("--torchtomo-commit", help="recorded when torchtomo is on the path without its git history")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ASTRA and LEAP need CUDA")
    device = torch.device("cuda")
    output = args.output or Path("libraries/results") / args.geometry
    libraries = args.libraries or available_libraries()
    metadata = benchmark_metadata()
    if args.torchtomo_commit:
        metadata["torchtomo_commit"] = args.torchtomo_commit
    results = {
        "geometry": args.geometry,
        "libraries": libraries,
        "device": torch.cuda.get_device_name(),
        "metadata": metadata,
        "started": datetime.now(timezone.utc).isoformat(),
        "settings": {key: str(value) for key, value in vars(args).items()},
    }
    warm()
    for case in cases_for(args, device):
        if "agreement" in args.sections:
            results.setdefault("agreement", []).extend(measure_agreement(case, libraries, device))
        if "performance" in args.sections:
            warm(1.0)
            results.setdefault("performance", []).extend(measure_performance(case, libraries, device))
        if "quality" in args.sections:
            if args.geometry == "dbt":
                quality = dbt_quality(case, libraries, args.sart_iterations, args.subsets, device)
            elif case.volume_shape[0] <= args.quality_max_size:
                quality = cone_quality(case, libraries, args.sirt_iterations, device)
            else:
                quality = []
            results.setdefault("quality", []).extend(quality)
        del case
        gc_and_sync()
        output.mkdir(parents=True, exist_ok=True)
        (output / "cone-comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    results["finished"] = datetime.now(timezone.utc).isoformat()
    (output / "cone-comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "tables.md").write_text(markdown_tables(results))
    print(markdown_tables(results))


if __name__ == "__main__":
    main()
