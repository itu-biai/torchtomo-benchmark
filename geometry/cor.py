"""Centre-of-rotation estimates for a measured fan-beam scan.

Two offset models are fitted by differentiating the projector:

- "detector": the detector slid sideways, the source on the axis. Sliding the
  measured sinogram by d bins is exactly this model, which is what makes an
  injected shift a clean test of it.
- "axis": the rotation axis off the source-to-detector line, which in the frame
  of the gantry is source and detector slid by the same amount.

Both minimise the reconstruction's own data consistency,
    L(u) = || A_u fbp_u(y) - y ||^2 / N,
and a dense grid search over the same L checks that the gradient found its
global minimum rather than a local one. The baselines are the parallel-beam
methods a lab would reach for, from algotom: Vo's sinogram metric on the first
half turn, and phase correlation of opposite projections.

Every estimate is reported in detector bins, positive where the torchtomo
detector_shift is positive.
"""

import time

import numpy as np
import torch

MODELS = ("detector", "axis")


def pose_for(base: torch.Tensor, shift_px: torch.Tensor, model: str) -> torch.Tensor:
    columns = [base[:, 0], base[:, 1] + shift_px]
    columns.append(base[:, 2] + shift_px if model == "axis" else base[:, 2])
    return torch.stack(columns, dim=1)


def consistency(projector, sinogram, base, shift_px, model):
    projector.pose = pose_for(base, shift_px, model)
    return (projector.forward(projector.fbp(sinogram)) - sinogram).pow(2).mean()


def gradient_estimate(scan, model="detector", steps=30, start_bins=0.0, backend="auto"):
    """L-BFGS on one scalar, in bins; returns (bins, loss, seconds, evaluations).

    The loss is divided by its value at the start so that a unit step is about a
    bin whatever the data's scale, which is what L-BFGS's first step assumes.
    """
    device = scan.sinogram.device
    projector = scan.projector(backend=backend).to(device)
    base = projector.pose.detach().clone()
    bins = torch.tensor(float(start_bins), dtype=base.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([bins], lr=1, max_iter=steps, line_search_fn="strong_wolfe")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        scale = consistency(projector, scan.sinogram, base, bins * scan.bin_px, model).item()
    evaluations = 1

    def closure():
        nonlocal evaluations
        evaluations += 1
        optimizer.zero_grad()
        loss = consistency(projector, scan.sinogram, base, bins * scan.bin_px, model) / scale
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final = consistency(projector, scan.sinogram, base, bins * scan.bin_px, model).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return bins.item(), final, time.perf_counter() - start, evaluations + 1


@torch.no_grad()
def grid_losses(scan, bins_grid, model="detector", backend="auto"):
    projector = scan.projector(backend=backend).to(scan.sinogram.device)
    base = projector.pose.detach().clone()
    return np.array(
        [consistency(projector, scan.sinogram, base, torch.tensor(b * scan.bin_px), model).item() for b in bins_grid]
    )


def grid_estimate(scan, model="detector", half_width_bins=12.0, coarse=1.0, fine=0.02, backend="auto"):
    """Coarse grid, then a fine grid around its best point; returns (bins, loss, seconds, evaluations)."""
    start = time.perf_counter()
    coarse_grid = np.arange(-half_width_bins, half_width_bins + coarse / 2, coarse)
    coarse_losses = grid_losses(scan, coarse_grid, model, backend)
    centre = coarse_grid[np.argmin(coarse_losses)]
    fine_grid = np.arange(centre - coarse, centre + coarse + fine / 2, fine)
    fine_losses = grid_losses(scan, fine_grid, model, backend)
    best = int(np.argmin(fine_losses))
    return fine_grid[best], fine_losses[best], time.perf_counter() - start, len(coarse_grid) + len(fine_grid)


def _half_turn(scan):
    """Views of the first 180 degrees, as a numpy [views, bins] sinogram."""
    angles = scan.angles.numpy()
    keep = angles - angles[0] < np.pi - 1e-9
    return scan.sinogram[0, 0].cpu().numpy()[keep].astype(np.float64)


def _opposite_pair(scan):
    angles = scan.angles.numpy()
    sinogram = scan.sinogram[0, 0].cpu().numpy().astype(np.float64)
    opposite = int(np.argmin(np.abs(angles - (angles[0] + np.pi))))
    return sinogram[0], sinogram[opposite]


def vo_estimate(scan, search_bins=24.0):
    """algotom.find_center_vo on the first half turn, as an offset from the detector centre in bins."""
    from algotom.prep.calculation import find_center_vo

    half = _half_turn(scan)
    middle = (scan.n_det - 1) / 2
    start = time.perf_counter()
    centre = find_center_vo(half, start=middle - search_bins, stop=middle + search_bins, step=0.25, ratio=0.5)
    return middle - centre, time.perf_counter() - start


def phase_correlation_estimate(scan):
    """algotom.find_center_based_phase_correlation on the 0 and 180 degree projections."""
    from algotom.prep.calculation import find_center_based_phase_correlation

    first, opposite = _opposite_pair(scan)
    middle = (scan.n_det - 1) / 2
    start = time.perf_counter()
    rows = 8
    centre = find_center_based_phase_correlation(np.tile(first, (rows, 1)), np.tile(opposite, (rows, 1)))
    return middle - centre, time.perf_counter() - start


def slide(sinogram: torch.Tensor, bins: float) -> torch.Tensor:
    """The sinogram a detector slid by `bins` would have measured, by linear interpolation.

    Sliding the detector by +d moves every bin's position by +d, so bin j now
    reads what used to fall at position j + d.
    """
    n_det = sinogram.shape[-1]
    positions = torch.arange(n_det, dtype=torch.float64, device=sinogram.device) + bins
    low = positions.floor().long()
    weight = (positions - low).to(sinogram.dtype)
    valid_low = (low >= 0) & (low < n_det)
    valid_high = (low + 1 >= 0) & (low + 1 < n_det)
    gathered_low = sinogram[..., low.clamp(0, n_det - 1)] * valid_low
    gathered_high = sinogram[..., (low + 1).clamp(0, n_det - 1)] * valid_high
    return gathered_low * (1 - weight) + gathered_high * weight
