"""Summarize repeated training seeds on fixed data and dose, without pooling slices.

python training/summarize_repeats.py training/results-512-0.4.0-seed-*
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_runs(directories):
    runs = []
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        if config.get("methodology_revision") != 2 or config.get("dose_protocol") != "fixed":
            raise ValueError(f"{directory}: requires current fixed-dose methodology")
        if not config.get("truth_sha256"):
            raise ValueError(f"{directory}: missing ground-truth fingerprint")
        runs.append((directory, config, metrics))
    if not runs:
        raise ValueError("no runs supplied")
    # Only initialization/shuffling and the operator implementation may differ.
    fields = (
        "image_size",
        "angles",
        "geometry",
        "src_dist",
        "det_dist",
        "n_det",
        "seed",
        "photons",
        "truth_sha256",
        "windows_sha256",
        "device",
        "unet_epochs",
        "lpd_epochs",
        "batch_size",
        "learning_rate",
        "models",
        "unet_width",
        "lpd_iterations",
        "lpd_memory",
        "lpd_width",
        "iradon_width",
        "iradon_learning_rate",
        "n2i_splits",
        "p2p_grid",
        "classical_methods",
        "sirt_iterations",
        "sirt_relaxation",
        "sart_sweeps",
        "sart_subsets",
        "sart_relaxations",
        "red_iterations",
        "red_step",
        "red_weights",
        "bm3d_sigmas",
        "software",
    )
    reference = runs[0][1]
    test_ids = runs[0][2]["ids"]
    methods = set(runs[0][2]["methods"])
    groups = defaultdict(list)
    for directory, config, metrics in runs:
        differences = [key for key in fields if config.get(key) != reference.get(key)]
        if differences or metrics["ids"] != test_ids or set(metrics["methods"]) != methods:
            raise ValueError(f"{directory}: incompatible data/protocol/metrics: {differences}")
        label = config["projector"] + (":" + config["backend"] if config["projector"] == "torchtomo" else "")
        groups[label].append((directory, config, metrics))
    seed_sets = []
    for label, group in groups.items():
        seeds = [config["training_seed"] for _, config, _ in group]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError(f"{label}: need at least two distinct training seeds")
        seed_sets.append(set(seeds))
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("backend comparisons require matching training seed sets")
    windowed = all("mean_window_psnr_db" in metrics["methods"][method] for _, _, metrics in runs for method in methods)
    metric = "mean_window_psnr_db" if windowed else "mean_roi_psnr_db"
    result = {
        "metric": metric,
        "uncertainty": "sample standard deviation of test means across training seeds",
        "photons": reference["photons"],
        "dataset_seed": reference["seed"],
        "backends": {},
    }
    for label, group in groups.items():
        group = sorted(group, key=lambda run: run[1]["training_seed"])
        rows = {}
        for method in sorted(methods):
            values = [metrics["methods"][method][metric] for _, _, metrics in group]
            rows[method] = {
                "mean_db": statistics.mean(values),
                "std_across_seeds_db": statistics.stdev(values),
                "per_seed_db": values,
            }
        result["backends"][label] = {
            "training_seeds": [config["training_seed"] for _, config, _ in group],
            "directories": [str(directory) for directory, _, _ in group],
            "methods": rows,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize_runs(args.directories)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
