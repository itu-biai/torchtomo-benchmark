"""Fig 3 numbers: per-view motion and angle recovery, summaries and paired tests.

Run from the repository root after make motion and stamping:
    python isbi/motion_stats.py
"""

import json
from pathlib import Path

import pandas as pd
from paired_stats import test, with_holm

RESULTS = Path(__file__).resolve().parent / "results"
METRICS: tuple[str, ...] = ("shift_rmse", "angle_rmse_degrees", "psnr")


def load(path: Path = RESULTS / "motion.json") -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text())["results"]["rows"])


def summary(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (scenario, method), block in frame.groupby(["scenario", "method"], sort=False):
        row = {"scenario": scenario, "method": method, "n": len(block)}
        for metric in METRICS + ("seconds",):
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_std"] = float(block[metric].std())
        rows.append(row)
    return rows


def paired(frame: pd.DataFrame, scenario: str, ours: str, baselines: tuple[str, ...]) -> list[dict]:
    block = frame[frame.scenario == scenario]
    rows = []
    for baseline in baselines:
        for metric in METRICS:
            table = block.pivot_table(index=["phantom", "seed"], columns="method", values=metric)
            sign = -1.0 if metric == "psnr" else 1.0  # test() expects lower is better
            rows.append(
                {
                    "scenario": scenario,
                    "ours": ours,
                    "baseline": baseline,
                    "metric": metric,
                    **test(sign * table[ours].to_numpy(), sign * table[baseline].to_numpy()),
                }
            )
    return with_holm(rows)


def main() -> None:
    frame = load()
    out = {
        "summary": summary(frame),
        "tests": {
            "shift+angle": paired(frame, "shift+angle", "gradient pose", ("matching", "gradient shift")),
            "shift": paired(frame, "shift", "gradient shift", ("matching",)),
        },
    }
    (RESULTS / "motion_table.json").write_text(json.dumps(out, indent=1))
    for row in out["summary"]:
        print(
            f"{row['scenario']:12s} {row['method']:15s} shift {row['shift_rmse_mean']:.3f}±{row['shift_rmse_std']:.3f} "
            f"angle {row['angle_rmse_degrees_mean']:.3f}±{row['angle_rmse_degrees_std']:.3f} "
            f"psnr {row['psnr_mean']:.2f}±{row['psnr_std']:.2f} {row['seconds_mean']:.2f} s"
        )
    for scenario, rows in out["tests"].items():
        for row in rows:
            print(
                f"{scenario:12s} {row['ours']} vs {row['baseline']:15s} {row['metric']:19s} "
                f"median diff {row['median_difference']:+.4f} p_holm {row['p_holm']:.2e} r {row['r']:+.2f}"
            )


if __name__ == "__main__":
    main()
