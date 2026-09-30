"""Centre of rotation on measured data: gradient descent against the classical methods.

Run from the repository root:
    python geometry/calibrate_real.py

For each scan it reports, in detector bins:

- the gradient estimate for the detector and the axis model,
- a dense grid search over the same objective, which says whether the gradient
  found the global minimum,
- Vo's method and phase correlation from algotom, which assume parallel beam,
- an injection test: the measured sinogram slid by a known d bins, which is
  exactly a detector moved by d, so every method should move its estimate by d.
  The error of that move is the one number here with a ground truth. The shifts
  are fractional so that no method's search step lands on them exactly.

Results go to geometry/results/cor.json and the FBPs before and after
calibration to geometry/results/fbp.npz, which git leaves out.
"""

import argparse
import json
import time
from pathlib import Path

import cor
import datasets
import numpy as np
import torch

RESULTS = Path(__file__).resolve().parent / "results"
INJECTED = (-5.3, -2.6, 2.6, 5.3)


def scans(img_size):
    for sample in datasets.HTC_SAMPLES:
        yield datasets.htc2022(sample, img_size=img_size, flip_detector=True)
    yield datasets.walnut(views=1200, img_size=img_size, flip_detector=True)
    sparse = datasets.walnut(views=120, img_size=img_size, flip_detector=True)
    sparse.name = "walnut_120"
    yield sparse


def estimates(scan, steps):
    result = {}
    for model in cor.MODELS:
        bins, loss, seconds, evaluations = cor.gradient_estimate(scan, model, steps=steps)
        result[f"gradient_{model}"] = dict(bins=bins, loss=loss, seconds=seconds, evaluations=evaluations)
        bins, loss, seconds, evaluations = cor.grid_estimate(scan, model)
        result[f"grid_{model}"] = dict(bins=bins, loss=loss, seconds=seconds, evaluations=evaluations)
    bins, seconds = cor.vo_estimate(scan)
    result["vo"] = dict(bins=bins, seconds=seconds)
    bins, seconds = cor.phase_correlation_estimate(scan)
    result["phase_correlation"] = dict(bins=bins, seconds=seconds)
    return result


def injection(scan, steps, reference):
    """Each method's estimate on the slid sinogram, minus its own estimate on the original."""
    measured = scan.sinogram
    moves = {name: [] for name in ("gradient_detector", "grid_detector", "vo", "phase_correlation")}
    for injected in INJECTED:
        scan.sinogram = cor.slide(measured, injected)
        moves["gradient_detector"].append(cor.gradient_estimate(scan, "detector", steps=steps)[0])
        moves["grid_detector"].append(cor.grid_estimate(scan, "detector")[0])
        moves["vo"].append(cor.vo_estimate(scan)[0])
        moves["phase_correlation"].append(cor.phase_correlation_estimate(scan)[0])
    scan.sinogram = measured
    return {
        name: dict(
            injected=list(INJECTED),
            moved=[value - reference[name]["bins"] for value in values],
            error=[value - reference[name]["bins"] - injected for value, injected in zip(values, INJECTED)],
        )
        for name, values in moves.items()
    }


@torch.no_grad()
def reconstructions(name, img_size, bins_by_model, device):
    loader = datasets.walnut if name.startswith("walnut") else datasets.htc2022
    kwargs = {"views": 120} if name == "walnut_120" else {}
    if not name.startswith("walnut"):
        kwargs["sample"] = name.removeprefix("htc2022_")
    scan = loader(img_size=img_size, flip_detector=True, **kwargs).to(device)
    images = {}
    for label, (model, bins) in bins_by_model.items():
        projector = scan.projector(backend="auto").to(device)
        base = projector.pose.detach().clone()
        projector.pose = cor.pose_for(base, torch.tensor(bins * scan.bin_px, dtype=base.dtype), model).to(device)
        images[label] = projector.fbp(scan.sinogram)[0, 0].cpu().numpy()
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256, help="grid the objective is evaluated on")
    parser.add_argument("--recon-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30, help="L-BFGS iterations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    RESULTS.mkdir(exist_ok=True)

    report, images = {}, {}
    for scan in scans(args.size):
        scan.to(device)
        start = time.perf_counter()
        result = estimates(scan, args.steps)
        result["injection"] = injection(scan, args.steps, result)
        result["geometry"] = dict(
            views=len(scan.angles), bins=scan.n_det, bin_mm=scan.bin_px * scan.pixel_mm, pixel_mm=scan.pixel_mm
        )
        report[scan.name] = result
        print(f"\n{scan.name}: {time.perf_counter() - start:.0f} s", flush=True)
        for method in ("gradient_detector", "grid_detector", "gradient_axis", "grid_axis", "vo", "phase_correlation"):
            entry = result[method]
            extra = f"  loss {entry['loss']:.5e}" if "loss" in entry else ""
            print(f"  {method:18s} {entry['bins']:+8.3f} bins  {entry['seconds']:6.2f} s{extra}", flush=True)
        for method, entry in result["injection"].items():
            errors = ", ".join(f"{error:+.3f}" for error in entry["error"])
            print(f"  injection {method:18s} errors [{errors}] bins", flush=True)

        chosen = {
            "nominal": ("detector", 0.0),
            "detector": ("detector", result["gradient_detector"]["bins"]),
            "axis": ("axis", result["gradient_axis"]["bins"]),
        }
        for label, image in reconstructions(scan.name, args.recon_size, chosen, device).items():
            images[f"{scan.name}/{label}"] = image
        del scan
        torch.cuda.empty_cache()

    (RESULTS / "cor.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(RESULTS / "fbp.npz", **images)


if __name__ == "__main__":
    main()
