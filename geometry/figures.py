"""Figures from calibrate_real.py's results: FBP before and after calibration.

Run from the repository root after calibrate_real.py:
    python geometry/figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def crop(image, fraction=0.4):
    size = image.shape[0]
    half = int(size * fraction / 2)
    centre = size // 2
    return image[centre - half : centre + half, centre - half : centre + half]


def main():
    images = np.load(RESULTS / "fbp.npz")
    report = json.loads((RESULTS / "cor.json").read_text())
    scans = ["walnut", "walnut_120", "htc2022_ta"]
    figure, axes = plt.subplots(len(scans), 3, figsize=(10.5, 3.6 * len(scans)))
    for row, scan in enumerate(scans):
        nominal = images[f"{scan}/nominal"]
        calibrated = images[f"{scan}/detector"]
        low, high = np.percentile(calibrated, [1, 99.7])
        bins = report[scan]["gradient_detector"]["bins"]
        panels = [
            (crop(nominal), f"{scan}: nominal geometry"),
            (crop(calibrated), f"calibrated, detector shift {bins:+.2f} bins"),
            (crop(calibrated - nominal), "difference"),
        ]
        for column, (image, title) in enumerate(panels):
            axis = axes[row, column]
            if column < 2:
                axis.imshow(image, cmap="gray", vmin=low, vmax=high)
            else:
                limit = np.abs(image).max()
                axis.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(RESULTS / "fbp_before_after.png", dpi=130)
    print(RESULTS / "fbp_before_after.png")


if __name__ == "__main__":
    main()
