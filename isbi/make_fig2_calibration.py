"""Fig 2: nominal against calibrated FBP on measured scans, with zoom insets.

Run from the repository root after make calibrate (needs geometry/results/fbp.npz):
    python isbi/make_fig2_calibration.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
SCANS: dict[str, str] = {"walnut": "walnut, 1200 views", "walnut_120": "walnut, 120 views", "htc2022_tb": "HTC 2022 tb"}


def crop(image: np.ndarray, fraction: float) -> np.ndarray:
    half, centre = int(image.shape[0] * fraction / 2), image.shape[0] // 2
    return image[centre - half : centre + half, centre - half : centre + half]


def plot(images: np.lib.npyio.NpzFile, cor: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, len(SCANS), figsize=(7.0, 4.9), constrained_layout=True)
    for column, (scan, title) in enumerate(SCANS.items()):
        nominal, calibrated = images[f"{scan}/nominal"], images[f"{scan}/detector"]
        low, high = np.percentile(calibrated, [1, 99.7])
        bins = cor[scan]["gradient_detector"]["bins"]
        for row, (image, label) in enumerate(((nominal, "nominal"), (calibrated, f"calibrated, {bins:+.2f} bins"))):
            axis = axes[row, column]
            axis.imshow(crop(image, 0.6), cmap="gray", vmin=low, vmax=high)
            inset = axis.inset_axes([0.62, 0.62, 0.36, 0.36])
            inset.imshow(crop(image, 0.18), cmap="gray", vmin=low, vmax=high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor("yellow")
            axis.set_title(f"{title}: {label}", fontsize=7)
            axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    cor = json.loads((RESULTS / "cor.json").read_text())["results"]
    plot(np.load(ROOT / "geometry" / "results" / "fbp.npz"), cor, RESULTS / "figures" / "fig2_calibration.pdf")
    print(RESULTS / "figures" / "fig2_calibration.pdf")
