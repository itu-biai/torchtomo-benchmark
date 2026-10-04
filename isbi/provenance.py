"""Provenance stamp for every result the ISBI paper uses.

Wraps a raw result JSON as {"provenance": ..., "results": ...}, recording the
torchtomo and benchmark commits, versions, device, command and time.
"""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

import torchtomo

ROOT = Path(__file__).resolve().parents[1]
TORCHTOMO = ROOT.parent / "torchtomo"


@dataclass(frozen=True)
class Provenance:
    torchtomo_version: str
    torchtomo_commit: str
    torchtomo_dirty: bool
    benchmark_commit: str
    benchmark_dirty: bool
    torch: str
    cuda: str
    device: str
    command: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


SOURCE = ("*.py", "*.sh", "Makefile", "requirements.txt")


def _dirty(repo: Path) -> bool:
    """Uncommitted or untracked source files; result files written by the runs do not count."""
    return bool(_git(repo, "status", "--porcelain", "--", *SOURCE))


def current(command: str) -> Provenance:
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return Provenance(
        torchtomo_version=torchtomo.__version__,
        torchtomo_commit=_git(TORCHTOMO, "rev-parse", "HEAD"),
        torchtomo_dirty=_dirty(TORCHTOMO),
        benchmark_commit=_git(ROOT, "rev-parse", "HEAD"),
        benchmark_dirty=_dirty(ROOT),
        torch=torch.__version__,
        cuda=str(torch.version.cuda),
        device=device,
        command=command,
    )


def stamp(source: Path, target: Path, provenance: Provenance) -> None:
    payload = json.loads(source.read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"provenance": asdict(provenance), "results": payload}, indent=1))


if __name__ == "__main__":
    # python isbi/provenance.py <source.json> <target.json> "<command that made source>"
    stamp(Path(sys.argv[1]), Path(sys.argv[2]), current(sys.argv[3]))
    print(sys.argv[2])
