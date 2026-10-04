"""Fig 3: per-case shift and angle RMSE, shifts-and-angles scenario.

Run from the repository root after isbi/motion_stats.py:
    python isbi/make_fig3_motion.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from motion_stats import RESULTS, load  # noqa: E402

ORDER: tuple[str, ...] = ("none", "matching", "gradient shift", "gradient pose")
LABELS: dict[str, str] = {
    "none": "none",
    "matching": "proj.\nmatching",
    "gradient shift": "grad.\nshifts",
    "gradient pose": "grad.\npose",
}


def plot(frame: pd.DataFrame, path: Path) -> None:
    block = frame[frame.scenario == "shift+angle"]
    figure, axes = plt.subplots(1, 2, figsize=(3.45, 1.9), constrained_layout=True)
    panels = (("shift_rmse", "shift RMSE (px)"), ("angle_rmse_degrees", "angle RMSE (deg)"))
    for axis, (metric, label) in zip(axes, panels):
        for x, method in enumerate(ORDER):
            values = block[block.method == method][metric]
            axis.scatter([x] * len(values), values, s=5, alpha=0.6, color=f"C{x}", linewidths=0)
            axis.hlines(values.mean(), x - 0.3, x + 0.3, color="black", linewidth=1)
        axis.set_xticks(range(len(ORDER)), [LABELS[m] for m in ORDER], fontsize=6)
        axis.set_yscale("log")
        axis.set_ylabel(label, fontsize=7)
        axis.tick_params(axis="y", labelsize=6)
    psnr = block.groupby("method")["psnr"].mean()
    caption = "  ".join(f"{LABELS[m].replace(chr(10), ' ')}: {psnr[m]:.2f} dB" for m in ORDER)
    figure.suptitle(f"mean PSNR, {caption}", fontsize=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    plot(load(), RESULTS / "figures" / "fig3_motion.pdf")
    print(RESULTS / "figures" / "fig3_motion.pdf")
