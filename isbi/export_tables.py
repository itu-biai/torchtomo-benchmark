"""Final Tables for the paper: Markdown for paper_plan.md and LaTeX for the PDF.

Run from the repository root after the stats scripts:
    python isbi/export_tables.py
"""

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
TABLES = RESULTS / "tables"
COST_COLUMNS: dict[str, str] = {
    "Thies et al. backprojection": "Thies et al. backprojection (ms)",
    "torchtomo torch backproject": "torchtomo PyTorch backproject (ms)",
    "torchtomo backproject": "torchtomo CUDA backproject (ms)",
    "torchtomo forward": "torchtomo CUDA forward (ms)",
    "torchtomo adjoint": "torchtomo CUDA adjoint (ms)",
}
METHOD_NAMES: dict[str, str] = {
    "gradient_detector": "gradient (ours)",
    "grid_detector": "grid",
    "vo": "Vo's method",
    "phase_correlation": "phase correlation",
}


def markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def latex(header: list[str], rows: list[list[str]]) -> str:
    def escape(cell: str) -> str:
        return cell.replace("±", "$\\pm$").replace("%", "\\%").replace("_", "\\_")

    lines = [
        "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(escape(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(escape(c) for c in row) + " \\\\" for row in rows]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def cost_cell(values: list[float | None]) -> str:
    if any(v is None for v in values):
        return "OOM"
    values = np.asarray(values)
    return f"{np.median(values):.2f} ({values.min():.2f} to {values.max():.2f})"


def table1a() -> tuple[list[str], list[list[str]]]:
    sessions = [json.loads(p.read_text())["results"] for p in sorted(RESULTS.glob("geometry_gradients_s*.json"))]
    cases = sorted({(c["size"], c["views"], c["bins"]) for c in sessions[0]["cases"]})
    rows = []
    for size, views, bins in cases:
        row = [f"{size}, {views}"]
        for method in COST_COLUMNS:
            values = [
                c["milliseconds"]
                for s in sessions
                for c in s["cases"]
                if (c["size"], c["views"], c["bins"], c["method"]) == (size, views, bins, method)
            ]
            row.append(cost_cell(values) if values else "-")
        rows.append(row)
    memory = ["peak MiB at 512, 360"]
    for method in COST_COLUMNS:
        peaks = [
            c["peak_mib"]
            for s in sessions
            for c in s["cases"]
            if (c["size"], c["views"], c["method"]) == (512, 360, method) and c["peak_mib"]
        ]
        memory.append("n/a" if method.startswith("Thies") or not peaks else f"{np.median(peaks):.0f}")
    return ["Size, views"] + list(COST_COLUMNS.values()), rows + [memory]


def table1b() -> tuple[list[str], list[list[str]]]:
    rows = json.loads((RESULTS / "gradient_accuracy.json").read_text())["results"]["rows"]
    out = []
    for operator in ("forward", "adjoint", "backproject", "fbp"):
        cells = [operator]
        for geometry in ("parallel", "fan"):
            for path in ("cuda32", "torch32"):
                errors = [
                    r[f"{path}_relative_error"] for r in rows if (r["geometry"], r["operator"]) == (geometry, operator)
                ]
                cells.append(f"{np.median(errors):.1e}")
        out.append(cells)
    header = ["Operator", "Parallel, CUDA f32", "Parallel, PyTorch f32", "Fan, CUDA f32", "Fan, PyTorch f32"]
    return header, out


def table2() -> tuple[list[str], list[list[str]]]:
    rows = json.loads((RESULTS / "cor_table.json").read_text())["rows"]
    return (
        ["Scans", "Method", "Median error (bins)", "Worst error (bins)", "Time (s)"],
        [
            [r["scans"], METHOD_NAMES[r["method"]], f"{r['median']:.3f}", f"{r['worst']:.3f}", f"{r['seconds']:.2f}"]
            for r in rows
        ],
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name, (header, rows) in {"table1a": table1a(), "table1b": table1b(), "table2": table2()}.items():
        (TABLES / f"{name}.tex").write_text(latex(header, rows))
        blocks.append(f"**{name}**\n\n" + markdown(header, rows))
    text = "\n\n".join(blocks) + "\n"
    (TABLES / "final_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
