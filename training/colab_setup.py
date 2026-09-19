"""Run with `colab exec -s torchtomo-512 -f training/colab_setup.py`.

Expects two uploads: /content/torchtomo-source.tar.gz, an archive of the torchtomo
repository, and /content/torchtomo-benchmark.tar.gz, an archive of this one.
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

library = Path("/content/torchtomo")
root = Path("/content/torchtomo-benchmark")
library.mkdir(exist_ok=True)
root.mkdir(exist_ok=True)

# Clear previously extracted sources so a renamed or deleted file cannot linger,
# while keeping any results directory produced by an earlier run.
for stale in (library / "src", library / "tests"):
    shutil.rmtree(stale, ignore_errors=True)
for folder in ("libraries", "tests", "training"):
    for stale in (root / folder).glob("*.py"):
        stale.unlink()

with tarfile.open("/content/torchtomo-source.tar.gz") as archive:
    archive.extractall(library, filter="data")
with tarfile.open("/content/torchtomo-benchmark.tar.gz") as archive:
    archive.extractall(root, filter="data")

# Optional packed CT slices, uploaded separately because they are far larger.
slices = Path("/content/ct-subset.tar")
if slices.exists():
    shutil.rmtree("/content/ct-subset", ignore_errors=True)
    with tarfile.open(slices, "r:*") as archive:
        archive.extractall("/content", filter="data")
    packed = sorted(Path("/content/ct-subset").glob("*.npz"))
    print("CT slices:", ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in packed), flush=True)

env = {**os.environ, "PYTHONPATH": str(library / "src")}
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires a CUDA Colab runtime")
print("GPU:", torch.cuda.get_device_name(), flush=True)


def run(command, cwd=root):
    """Print child output explicitly; Jupyter does not capture subprocess file descriptors."""
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    print("$", " ".join(command[1:]), flush=True)
    print(result.stdout[-4000:] or result.stderr[-4000:], flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {result.stderr[-2000:]}")


run([sys.executable, "-m", "pytest", "-q"], cwd=library)
run([sys.executable, "-m", "pytest", "-q"])
for batch_size in (1, 5):
    run([sys.executable, "training/profile_models.py", "--device", "cuda", "--batch-size", str(batch_size)])
