"""Record the software and code behind a result, including editable checkouts."""

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

import torchtomo


def benchmark_metadata():
    source = Path(torchtomo.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if path.suffix in {".py", ".cu"}:
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    benchmark_digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "libraries", root / "training"):
        for path in sorted(directory.glob("*.py")):
            benchmark_digest.update(str(path.relative_to(root)).encode())
            benchmark_digest.update(path.read_bytes())
    versions = {}
    for name in ("torchtomo", "numpy", "scikit-image", "leapct", "torch-radon", "astra-toolbox", "pytigre"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        commit = None  # A wheel installation does not require git.
    return {
        "benchmark_source_sha256": benchmark_digest.hexdigest(),
        "torchtomo_version": torchtomo.__version__,
        "torchtomo_source_sha256": digest.hexdigest(),
        "torchtomo_commit": commit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "packages": versions,
    }
