"""TIGRE-backed drop-ins for torchtomo's ParallelBeam and FanBeam.

TIGRE (CERN and the University of Bath) is a 3D cone-beam toolbox with a NumPy
interface: every call copies its input to the card and its result back, and that
round trip is part of what a TIGRE user pays, so it stays in the timings.

A 2D problem is a 3D one a single voxel thick. Parallel beam stacks the batch as
slices, which a parallel geometry keeps independent. Fan beam is a cone with one
detector row, which is exactly a fan, and runs one image at a time because a cone
would mix stacked slices. The adjoint is TIGRE's "matched" backprojector, which
TIGRE does not scale as the adjoint of its own forward projector: <Ax, y> and
<x, A^T y> differ by the pixel width in parallel beam and by about 0.49 in fan
beam. The adapter multiplies it by that ratio, measured once on TIGRE's own Ax,
so the pair it trains with is matched. FBP is
TIGRE's fbp for parallel beam and fdk (the one-row cone is the fan-beam FBP) for
fan beam, both with the Ram-Lak filter.

TIGRE starts its gantry a quarter turn on from torchtomo's and reads its detector
the other way; in parallel beam it also turns the other way round. With those
fixed and everything on torchtomo's [-1, 1] image, the forward projectors agree
with torchtomo's with no fitted scale.

TIGRE is an optional dependency, built from github.com/CERN/TIGRE with
patches/tigre-texture-copy-sync.patch applied: without it the forward projectors
can read the image before its copy to the card has finished.
"""

import numpy as np
import tigre
import torch
from matched_pair import MatchedPairMixin
from tigre.algorithms import fbp as tigre_fbp
from tigre.algorithms import fdk as tigre_fdk
from tigre.utilities.Atb import Atb
from tigre.utilities.Ax import Ax

from torchtomo import FanBeam, ParallelBeam

# TIGRE's interpolated projector samples the whole source-to-detector segment, so
# a parallel "source" far away costs samples in proportion to its distance. Two
# units clears the image's corners on [-1, 1].
PARALLEL_SOURCE_DISTANCE = 2.0


def _geometry(mode, size, slices, n_det, det_spacing, source_distance, detector_distance):
    pixel = 2.0 / size
    geometry = tigre.geometry()
    geometry.mode = mode
    geometry.DSO = source_distance
    geometry.DSD = source_distance + detector_distance
    geometry.nVoxel = np.array([slices, size, size])
    geometry.dVoxel = np.array([pixel, pixel, pixel])
    geometry.sVoxel = geometry.nVoxel * geometry.dVoxel
    geometry.nDetector = np.array([slices, n_det])
    geometry.dDetector = np.array([pixel, det_spacing])
    geometry.sDetector = geometry.nDetector * geometry.dDetector
    geometry.offOrigin = np.zeros(3)
    geometry.offDetector = np.zeros(2)
    geometry.rotDetector = np.zeros(3)
    geometry.accuracy = 0.5
    geometry.COR = 0
    return geometry


def _leave_tigre_half_the_card():
    """TIGRE refuses to run while less than half the card is free, as it would with
    another program on it. PyTorch's cache counts against that, so hand it back."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if free < total / 2:
            torch.cuda.empty_cache()


class _TigreBase(MatchedPairMixin):
    def tigre_angles(self):
        return np.ascontiguousarray(
            self.angle_direction * self.angles.detach().cpu().numpy() + np.pi / 2, dtype=np.float32
        )

    def to_tigre(self, sinogram):
        """[B, angles, detectors] torch -> [angles, B, detectors] NumPy, detector reversed."""
        return np.ascontiguousarray(sinogram.detach().flip(-1).permute(1, 0, 2).cpu().numpy(), dtype=np.float32)

    def from_tigre(self, sinogram, device):
        return torch.from_numpy(np.ascontiguousarray(sinogram)).to(device).permute(1, 0, 2).flip(-1)

    @torch.no_grad()
    def project_raw(self, image):
        _leave_tigre_half_the_card()
        slabs = self.slabs(image.shape[0])
        images = image[:, 0].detach().cpu().numpy().astype(np.float32)
        views = [
            Ax(np.ascontiguousarray(images[s]), self.geometry(len(images[s])), self.tigre_angles(), "interpolated")
            for s in slabs
        ]
        return self.from_tigre(np.concatenate(views, axis=1), image.device).unsqueeze(1).contiguous()

    @torch.no_grad()
    def backproject_raw(self, sinogram):
        return self.tigre_backproject(sinogram) * self.adjoint_scale()

    def adjoint_scale(self):
        """The constant that makes TIGRE's backprojector the adjoint of its Ax."""
        if getattr(self, "_adjoint_scale", None) is None:
            generator = torch.Generator().manual_seed(0)
            image = torch.rand(1, 1, self.img_size, self.img_size, generator=generator)
            sinogram = torch.rand(1, 1, self.n_angles, self.n_det, generator=generator)
            forward = (self.project_raw(image) * sinogram).sum()
            self._adjoint_scale = float(forward / (image * self.tigre_backproject(sinogram)).sum())
        return self._adjoint_scale

    @torch.no_grad()
    def tigre_backproject(self, sinogram):
        _leave_tigre_half_the_card()
        views = self.to_tigre(sinogram[:, 0])
        slabs = self.slabs(sinogram.shape[0])
        images = [
            Atb(np.ascontiguousarray(views[:, s]), self.geometry(views[:, s].shape[1]), self.tigre_angles(), "matched")
            for s in slabs
        ]
        return torch.from_numpy(np.concatenate(images)).to(sinogram.device).unsqueeze(1)

    @torch.no_grad()
    def fbp(self, sinogram, filter_name="ramp"):
        _leave_tigre_half_the_card()
        if filter_name != "ramp":
            raise NotImplementedError(f"the TIGRE adapter offers the ramp filter only, not {filter_name!r}")
        views = self.to_tigre(sinogram[:, 0])
        slabs = self.slabs(sinogram.shape[0])
        images = [
            self.reconstruct(
                np.ascontiguousarray(views[:, s]),
                self.geometry(views[:, s].shape[1]),
                self.tigre_angles(),
                filter="ram_lak",
            )
            for s in slabs
        ]
        return self.masked(torch.from_numpy(np.concatenate(images)).to(sinogram.device).unsqueeze(1))


class TigreParallelBeam(_TigreBase, ParallelBeam):
    """ParallelBeam with TIGRE's CUDA kernels; the batch goes through as one stack of slices."""

    reconstruct = staticmethod(tigre_fbp)
    angle_direction = -1.0

    def __init__(
        self, img_size=256, n_angles=180, n_det=None, angle_range=(0, np.pi), circle=True, angles=None, **kwargs
    ):
        super().__init__(img_size, n_angles, n_det, angle_range, circle, angles=angles, **kwargs)
        if self.n_det != img_size:
            raise ValueError("the TIGRE geometry here assumes a detector as wide as the image")

    def slabs(self, batch):
        return [slice(0, batch)]

    def geometry(self, slices):
        pixel = 2.0 / self.img_size
        return _geometry(
            "parallel", self.img_size, slices, self.n_det, pixel, PARALLEL_SOURCE_DISTANCE, PARALLEL_SOURCE_DISTANCE
        )


class TigreFanBeam(_TigreBase, FanBeam):
    """FanBeam as TIGRE's one-row cone beam, one image at a time."""

    reconstruct = staticmethod(tigre_fdk)
    angle_direction = 1.0

    def __init__(
        self,
        img_size=256,
        n_angles=360,
        n_det=None,
        src_dist=None,
        det_dist=None,
        det_width=None,
        det_spacing=None,
        angle_range=(0, 2 * np.pi),
        n_samples=None,
        circle=True,
        angles=None,
    ):
        super().__init__(
            img_size,
            n_angles,
            n_det,
            src_dist,
            det_dist,
            det_width,
            det_spacing,
            angle_range,
            n_samples,
            circle,
            angles=angles,
        )

    def slabs(self, batch):
        return [slice(index, index + 1) for index in range(batch)]

    def geometry(self, slices):
        pixel = 2.0 / self.img_size
        return _geometry(
            "cone",
            self.img_size,
            slices,
            self.n_det,
            self.det_width / self.n_det * pixel,
            self.src_dist * pixel,
            self.det_dist * pixel,
        )
