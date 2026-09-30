"""Measured fan-beam sinograms from public datasets, in torchtomo's units.

Every loader returns a `Scan`: a sinogram [1, 1, n_angles, n_det] of line
integrals and the geometry that measured it, with every length in pixels of the
reconstruction grid. The reconstruction's field of view is the detector's width
projected back to the axis, so the inscribed circle holds everything the
detector saw.

Files are fetched from Zenodo on first use into `geometry/data/`:

- HTC 2022 (Meaney et al., Zenodo 6984868): five 2D fan-beam scans, 721 views
  over 360 degrees, 560 bins of 0.2 mm, line integrals already.
- Walnut (Hamalainen et al. 2015, Zenodo 1254206): 1200 views over 360 degrees,
  2296 raw bins of 0.05 mm with no documented centre-of-rotation correction.
"""

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch

from torchtomo import FanBeam

DATA = Path(__file__).resolve().parent / "data"
ZENODO = "https://zenodo.org/api/records/{record}/files/{name}/content"
HTC_SAMPLES = ("solid_disc", "ta", "tb", "tc", "td")


@dataclass
class Scan:
    name: str
    sinogram: torch.Tensor
    angles: torch.Tensor
    src_dist: float
    det_dist: float
    det_width: float
    img_size: int
    pixel_mm: float

    @property
    def n_det(self) -> int:
        return self.sinogram.shape[-1]

    @property
    def bin_px(self) -> float:
        """One detector bin in pixels of the reconstruction grid, at the detector."""
        return self.det_width / self.n_det

    def projector(self, **kwargs) -> FanBeam:
        return FanBeam(
            img_size=self.img_size,
            n_angles=len(self.angles),
            n_det=self.n_det,
            src_dist=self.src_dist,
            det_dist=self.det_dist,
            det_width=self.det_width,
            angles=self.angles,
            **kwargs,
        )

    def to(self, device) -> "Scan":
        self.sinogram = self.sinogram.to(device)
        return self


def _fetch(record: int, name: str, local: str) -> Path:
    path = DATA / local
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ZENODO.format(record=record, name=name), path)
    return path


def _scan(name, sinogram, angles, sod_mm, sdd_mm, det_width_mm, img_size, flip_detector, reverse_angles):
    """Express a scan in pixels of a grid whose inscribed circle is the field of view."""
    magnification = sdd_mm / sod_mm
    pixel_mm = det_width_mm / magnification / img_size
    if flip_detector:
        sinogram = sinogram[:, ::-1]
    if reverse_angles:
        angles = -angles
    return Scan(
        name=name,
        sinogram=torch.from_numpy(np.ascontiguousarray(sinogram, dtype=np.float32)).view(1, 1, *sinogram.shape),
        angles=torch.as_tensor(angles, dtype=torch.float64),
        src_dist=sod_mm / pixel_mm,
        det_dist=(sdd_mm - sod_mm) / pixel_mm,
        det_width=det_width_mm / pixel_mm,
        img_size=img_size,
        pixel_mm=pixel_mm,
    )


def htc2022(sample: str = "ta", img_size: int = 512, flip_detector=False, reverse_angles=False) -> Scan:
    path = _fetch(6984868, f"htc2022_{sample}_full.mat", f"htc2022_{sample}_full.mat")
    data = scipy.io.loadmat(path)["CtDataFull"][0, 0]
    params = data["parameters"][0, 0]
    sinogram = data["sinogram"].astype(np.float64)
    angles = np.deg2rad(params["angles"].ravel())
    sod = float(params["distanceSourceOrigin"].item())
    sdd = float(params["distanceSourceDetector"].item())
    width = float(params["pixelSizePost"].item()) * sinogram.shape[1]
    return _scan(f"htc2022_{sample}", sinogram, angles, sod, sdd, width, img_size, flip_detector, reverse_angles)


def htc2022_reference(sample: str = "ta") -> np.ndarray:
    """The organisers' 512 px FBP of the full data."""
    path = _fetch(6984868, f"htc2022_{sample}_full_recon_fbp.mat", f"htc2022_{sample}_full_recon_fbp.mat")
    return scipy.io.loadmat(path)["reconFullFbp"]


def walnut(views: int = 1200, binning: int = 4, img_size: int = 512, flip_detector=False, reverse_angles=False) -> Scan:
    """The raw walnut counts, flat-fielded from the air at both detector edges and binned."""
    path = _fetch(1254206, "FullSizeSinograms.mat", "walnut_full.mat")
    counts = scipy.io.loadmat(path)[f"sinogram{views}"].astype(np.float64).T
    n_det = counts.shape[1] // binning * binning
    counts = counts[:, :n_det].reshape(counts.shape[0], -1, binning).mean(axis=2)
    edge = max(4, counts.shape[1] // 40)
    air = np.concatenate([counts[:, :edge], counts[:, -edge:]], axis=1).mean(axis=1, keepdims=True)
    sinogram = np.log(air / np.clip(counts, 1.0, None))
    step = 2 * math.pi / views
    angles = np.arange(views) * step
    width = 0.05 * n_det
    return _scan("walnut", sinogram, angles, 110.0, 300.0, width, img_size, flip_detector, reverse_angles)
