"""Table 2: centre-of-rotation injection errors, times and paired tests.

Run from the repository root after make calibrate and stamping:
    python isbi/cor_stats.py
"""

import json
from pathlib import Path

import numpy as np
from paired_stats import test, with_holm

RESULTS = Path(__file__).resolve().parent / "results"
METHODS: tuple[str, ...] = ("gradient_detector", "grid_detector", "vo", "phase_correlation")
GROUPS: dict[str, tuple[str, ...]] = {
    "HTC 2022 (5)": ("htc2022_solid_disc", "htc2022_ta", "htc2022_tb", "htc2022_tc", "htc2022_td"),
    "walnut, 1200 views": ("walnut",),
    "walnut, 120 views": ("walnut_120",),
}


def injection_errors(cor: dict, scans: tuple[str, ...], method: str) -> np.ndarray:
    return np.abs(np.concatenate([cor[scan]["injection"][method]["error"] for scan in scans]))


def table(cor: dict) -> list[dict]:
    rows = []
    for label, scans in GROUPS.items():
        for method in METHODS:
            errors = injection_errors(cor, scans, method)
            rows.append(
                {
                    "scans": label,
                    "method": method,
                    "median": float(np.median(errors)),
                    "worst": float(errors.max()),
                    "seconds": float(np.mean([cor[scan][method]["seconds"] for scan in scans])),
                }
            )
    return rows


def models(cor: dict) -> list[dict]:
    rows = []
    for scan, entry in cor.items():
        detector, axis = entry["gradient_detector"]["bins"], entry["gradient_axis"]["bins"]
        rows.append(
            {"scan": scan, "detector_bins": detector, "axis_bins": axis, "ratio": detector / axis if axis else None}
        )
    return rows


def main() -> None:
    cor = json.loads((RESULTS / "cor.json").read_text())["results"]
    every = tuple(scan for scans in GROUPS.values() for scan in scans)
    ours = injection_errors(cor, every, "gradient_detector")
    tests = with_holm([{"baseline": m, **test(ours, injection_errors(cor, every, m))} for m in METHODS[1:]])
    estimates = {scan: {m: cor[scan][m]["bins"] for m in METHODS + ("gradient_axis",)} for scan in cor}
    out = {"rows": table(cor), "tests": tests, "models": models(cor), "estimates": estimates}
    (RESULTS / "cor_table.json").write_text(json.dumps(out, indent=1))
    for row in out["rows"]:
        print(
            f"{row['scans']:20s} {row['method']:18s} median {row['median']:8.3f} worst {row['worst']:8.3f} "
            f"{row['seconds']:6.2f} s"
        )
    for row in tests:
        print(
            f"gradient vs {row['baseline']:18s} n {row['n']} median diff {row['median_difference']:+.3f} "
            f"p_holm {row['p_holm']:.2e} r {row['r']:+.2f}"
        )


if __name__ == "__main__":
    main()
