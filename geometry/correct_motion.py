"""Per-view pose errors, recovered by differentiating the pose table.

Run from the repository root:
    python geometry/correct_motion.py

The object drifts and jitters in the plane while it is scanned, and in the
second scenario the rotation stage also misreports each view's angle. In
parallel beam a translation (tx, ty) at view angle theta moves that view's
projection sideways by tx cos(theta) + ty sin(theta), which is the pose table's
detector_shift column; an angle error is its angle column. Every view has its
own unknowns.

Estimators see only the sinogram:

- "matching": projection matching, the classical method (Dengler 1989; Gursoy et
  al. 2017): reconstruct, reproject, register every measured view to its
  reprojection by subpixel cross-correlation, repeat. It models shifts only.
- "gradient shift": L-BFGS on all per-view shifts at once, minimising the
  reconstruction's own data consistency || A_u fbp_u(y) - y ||^2, coarse to
  fine over sinograms blurred along the detector, since noise gives the
  objective local minima.
- "gradient pose": the same objective over shifts and angles together. Nothing
  but the parametrisation changes.

What no method can see is removed before errors are measured: a rigid
translation of the object, u = tx cos + ty sin, and a constant angle offset,
which is a rotation of it. Reconstructions are made with each estimate moved
onto the truth's invisible component, so image metrics compare one placement.

Phantoms are the HTC 2022 organisers' reconstructions of five measured objects
(datasets.py), resampled into the inscribed circle, plus Shepp-Logan.
"""

import argparse
import json
import time
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F

from torchtomo import ParallelBeam, shepp_logan

RESULTS = Path(__file__).resolve().parent / "results"
SCENARIOS = {"shift": 0.0, "shift+angle": None}


def phantoms(size):
    yield "shepp_logan", shepp_logan(size)
    for sample in datasets.HTC_SAMPLES:
        image = torch.from_numpy(datasets.htc2022_reference(sample)).float()
        image = F.interpolate(image.view(1, 1, *image.shape), size=(size, size), mode="area")[0, 0]
        yield f"htc2022_{sample}", (image / image.max()).clamp(min=0)


def true_shifts(angles, amplitude, jitter, generator):
    """Smooth drift in x and y plus per-view jitter, as detector shifts in pixels."""
    t = (angles - angles[0]) / (angles[-1] - angles[0])
    drift = []
    for _ in range(2):
        path = torch.zeros_like(t)
        for harmonic in range(1, 4):
            phase = 2 * np.pi * torch.rand((), generator=generator, dtype=t.dtype)
            path = path + torch.randn((), generator=generator, dtype=t.dtype) / harmonic * torch.sin(
                harmonic * np.pi * t + phase
            )
        drift.append(amplitude * path / path.abs().max())
    tx, ty = drift
    shifts = tx * torch.cos(angles) + ty * torch.sin(angles)
    return shifts + jitter * torch.randn(len(angles), generator=generator, dtype=t.dtype)


def true_angle_errors(angles, jitter_degrees, generator):
    return np.deg2rad(jitter_degrees) * torch.randn(len(angles), generator=generator, dtype=angles.dtype)


class Invisible:
    """What a rigid motion of the object does to the pose: translation and rotation."""

    def __init__(self, angles):
        basis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.translation, _ = torch.linalg.qr(basis)

    def shifts(self, shifts):
        return self.translation @ (self.translation.T @ shifts)

    @staticmethod
    def angles(errors):
        return errors.mean() * torch.ones_like(errors)


def pose_with(base, shifts, angle_errors=None):
    angles = base[:, 0] if angle_errors is None else base[:, 0] + angle_errors
    return torch.stack([angles, base[:, 1] + shifts], dim=1)


def blur(sinogram, sigma_bins):
    """Gaussian blur along the detector; the coarse levels of a coarse-to-fine fit."""
    if sigma_bins == 0:
        return sinogram
    radius = int(3 * sigma_bins) + 1
    offsets = torch.arange(-radius, radius + 1, dtype=sinogram.dtype, device=sinogram.device)
    kernel = torch.exp(-(offsets**2) / (2 * sigma_bins**2))
    return F.conv2d(sinogram, (kernel / kernel.sum()).view(1, 1, 1, -1), padding=(0, radius))


def consistency(projector, sinogram, pose, sigma_bins=0):
    projector.pose = pose
    return (blur(projector.forward(projector.fbp(sinogram)), sigma_bins) - blur(sinogram, sigma_bins)).pow(2).mean()


def gradient_estimate(projector, sinogram, iterations, smoothness, fit_angles, sigmas=(4, 2, 1, 0)):
    """L-BFGS on per-view shifts, and angles when asked; returns (shifts, angle errors, evaluations).

    The objective has local minima once the data are noisy, so it is fitted
    coarse to fine: against sinograms blurred along the detector by each of
    `sigmas` bins in turn, each level starting where the last one stopped.

    Angles are optimised as the arc they sweep at the edge of the field of view,
    in pixels, so that both columns move the data by comparable amounts per unit.
    """
    base = projector.pose.detach().clone()
    views = len(base)
    radius = projector.img_size / 2
    shifts = torch.zeros(views, dtype=base.dtype, device=base.device, requires_grad=True)
    arcs = torch.zeros(views, dtype=base.dtype, device=base.device, requires_grad=fit_angles)
    parameters = [shifts, arcs] if fit_angles else [shifts]
    scale = sinogram.pow(2).mean().item()
    evaluations = 0

    def second_difference(values):
        return (values[2:] - 2 * values[1:-1] + values[:-2]).pow(2).mean()

    for sigma in sigmas:
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1,
            max_iter=iterations,
            history_size=20,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal evaluations
            evaluations += 1
            optimizer.zero_grad()
            pose = pose_with(base, shifts, arcs / radius)
            loss = consistency(projector, sinogram, pose, sigma) / scale
            loss = loss + smoothness * (second_difference(shifts) + second_difference(arcs))
            loss.backward()
            return loss

        optimizer.step(closure)
    projector.pose = base
    return shifts.detach(), (arcs / radius).detach(), evaluations


def _subpixel_lag(measured, reference, max_lag):
    """Lag in bins that best aligns each measured row to its reference row, by FFT cross-correlation."""
    n = measured.shape[-1]
    size = 2 * n
    spectrum = torch.fft.rfft(reference, size) * torch.conj(torch.fft.rfft(measured, size))
    correlation = torch.fft.irfft(spectrum, size)
    lags = torch.cat([torch.arange(0, max_lag + 1), torch.arange(-max_lag, 0)]).to(measured.device)
    window = correlation[:, lags % size]
    best = window.argmax(dim=1)
    left = window.gather(1, ((best - 1) % len(lags)).unsqueeze(1)).squeeze(1)
    centre = window.gather(1, best.unsqueeze(1)).squeeze(1)
    right = window.gather(1, ((best + 1) % len(lags)).unsqueeze(1)).squeeze(1)
    denominator = left - 2 * centre + right
    offset = torch.where(denominator.abs() > 1e-12, 0.5 * (left - right) / denominator, torch.zeros_like(centre))
    return lags[best].to(measured.dtype) + offset.clamp(-0.5, 0.5)


@torch.no_grad()
def matching_estimate(projector, sinogram, iterations, max_lag):
    base = projector.pose.detach().clone()
    shifts = torch.zeros(len(base), dtype=base.dtype, device=base.device)
    measured = sinogram[0, 0].double()
    for _ in range(iterations):
        projector.pose = pose_with(base, shifts)
        image = projector.fbp(sinogram)
        projector.pose = base
        reprojection = projector.forward(image)[0, 0].double()
        # A view whose detector moved by +s reads the reprojection at bin + s.
        shifts = -_subpixel_lag(reprojection, measured, max_lag).to(base.dtype)
    return shifts


def psnr(image, truth):
    mse = (image - truth).pow(2).mean()
    return (10 * torch.log10(truth.max() ** 2 / mse)).item()


@torch.no_grad()
def reconstruct(projector, sinogram, pose):
    base = projector.pose.detach().clone()
    projector.pose = pose
    image = projector.fbp(sinogram)
    projector.pose = base
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--angles", type=int, default=256)
    parser.add_argument("--amplitude", type=float, default=4.0, help="peak drift in pixels")
    parser.add_argument("--jitter", type=float, default=0.5, help="per-view shift jitter std in pixels")
    parser.add_argument("--angle-jitter", type=float, default=0.3, help="per-view angle error std in degrees")
    parser.add_argument("--noise", type=float, default=0.01, help="Gaussian noise relative to the sinogram's std")
    parser.add_argument("--iterations", type=int, default=60, help="L-BFGS iterations per level")
    parser.add_argument("--smoothness", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=100, help="projection matching rounds")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    RESULTS.mkdir(exist_ok=True)

    projector = ParallelBeam(img_size=args.size, n_angles=args.angles, circle=True, backend="auto").to(device)
    base = projector.pose.detach().clone()
    angles = base[:, 0]
    invisible = Invisible(angles)
    zeros = torch.zeros_like(angles)

    rows, images = [], {}
    for scenario, angle_jitter in SCENARIOS.items():
        angle_jitter = args.angle_jitter if angle_jitter is None else angle_jitter
        for name, phantom in phantoms(args.size):
            truth_image = phantom.view(1, 1, args.size, args.size).to(device)
            for seed in range(args.seeds):
                generator = torch.Generator().manual_seed(seed)
                shifts = true_shifts(angles.cpu(), args.amplitude, args.jitter, generator).to(device)
                angle_errors = true_angle_errors(angles.cpu(), angle_jitter, generator).to(device)
                with torch.no_grad():
                    projector.pose = pose_with(base, shifts, angle_errors)
                    sinogram = projector.forward(truth_image)
                    projector.pose = base
                    noise = torch.randn(sinogram.shape, generator=generator).to(device)
                    sinogram = sinogram + args.noise * sinogram.std() * noise

                estimates = {"none": (zeros, zeros, 0.0, 0)}
                for method in ("matching", "gradient shift", "gradient pose"):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    if method == "matching":
                        found = matching_estimate(projector, sinogram, args.rounds, max_lag=args.size // 8)
                        found, evaluations = (found, zeros), args.rounds
                    else:
                        *found, evaluations = gradient_estimate(
                            projector, sinogram, args.iterations, args.smoothness, fit_angles=method == "gradient pose"
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    estimates[method] = (*found, time.perf_counter() - start, evaluations)
                estimates["oracle"] = (shifts, angle_errors, 0.0, 0)

                for method, (found_shifts, found_angles, seconds, evaluations) in estimates.items():
                    placed_shifts = found_shifts - invisible.shifts(found_shifts) + invisible.shifts(shifts)
                    placed_angles = found_angles - invisible.angles(found_angles) + invisible.angles(angle_errors)
                    image = reconstruct(projector, sinogram, pose_with(base, placed_shifts, placed_angles))
                    shift_error = placed_shifts - shifts
                    angle_error = placed_angles - angle_errors
                    rows.append(
                        dict(
                            scenario=scenario,
                            phantom=name,
                            seed=seed,
                            method=method,
                            shift_rmse=shift_error.pow(2).mean().sqrt().item(),
                            angle_rmse_degrees=np.rad2deg(angle_error.pow(2).mean().sqrt().item()),
                            psnr=psnr(image.clamp(min=0), truth_image),
                            seconds=seconds,
                            evaluations=evaluations,
                        )
                    )
                    if seed == 0:
                        images[f"{scenario}/{name}/{method}"] = image[0, 0].cpu().numpy()
                        images[f"{scenario}/{name}/{method}/shifts"] = placed_shifts.cpu().numpy()
                        images[f"{scenario}/{name}/{method}/angles"] = placed_angles.cpu().numpy()
                if seed == 0:
                    images[f"{scenario}/{name}/truth"] = truth_image[0, 0].cpu().numpy()
                summary = "  ".join(
                    f"{r['method']} {r['shift_rmse']:.3f}px {r['angle_rmse_degrees']:.3f}deg {r['psnr']:.2f}dB"
                    for r in rows[-len(estimates) :]
                )
                print(f"{scenario} {name} seed {seed}: {summary}", flush=True)

    for scenario in SCENARIOS:
        print(f"\n{scenario}: mean over phantoms and seeds")
        for method in ("none", "matching", "gradient shift", "gradient pose", "oracle"):
            chosen = [r for r in rows if r["method"] == method and r["scenario"] == scenario]
            print(
                f"  {method:15s} shift rmse {np.mean([r['shift_rmse'] for r in chosen]):.3f} px  "
                f"angle rmse {np.mean([r['angle_rmse_degrees'] for r in chosen]):.3f} deg  "
                f"psnr {np.mean([r['psnr'] for r in chosen]):.2f} dB  "
                f"{np.mean([r['seconds'] for r in chosen]):.1f} s, "
                f"{np.mean([r['evaluations'] for r in chosen]):.0f} evals"
            )
    (RESULTS / "motion.json").write_text(json.dumps(dict(args=vars(args), rows=rows), indent=1))
    np.savez_compressed(RESULTS / "motion.npz", **images)


if __name__ == "__main__":
    main()
