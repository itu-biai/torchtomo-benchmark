"""Paired Wilcoxon signed-rank tests with Holm correction and rank-biserial r."""

import numpy as np
from scipy.stats import rankdata, wilcoxon


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def test(ours: np.ndarray, theirs: np.ndarray) -> dict[str, float]:
    """Lower is better for both; positive differences and r mean ours is lower."""
    differences = np.asarray(theirs, dtype=float) - np.asarray(ours, dtype=float)
    p_raw = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
    return {
        "n": int(len(differences)),
        "median_difference": float(np.median(differences)),
        "p_raw": p_raw,
        "r": rank_biserial(differences),
    }


def with_holm(rows: list[dict]) -> list[dict]:
    for row, adjusted in zip(rows, holm_adjust([row["p_raw"] for row in rows])):
        row["p_holm"] = adjusted
    return rows
