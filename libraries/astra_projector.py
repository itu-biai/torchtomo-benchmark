"""ASTRA-backed drop-ins for torchtomo's ParallelBeam and FanBeam.

As with leap_projector.py, subclassing keeps torchtomo's geometry, circle mask and
angle list, and replaces only the operators. Every call hands ASTRA the torch
tensors themselves through DLPack, so nothing is copied to the host.

Parallel beam uses ASTRA's 3D parallel projector with one detector row per image,
the batched route tomosipo takes. Fan beam has no batched 3D equivalent (a cone
would mix the slices), so each image goes through ASTRA's 2D fan projector in
turn. FBP is ASTRA's own FBP_CUDA algorithm, Ram-Lak filter, on linked GPU buffers.

The conventions were found by projecting the same phantom both ways: ASTRA turns
the gantry the other way round, and its 2D volumes run bottom to top, so the 2D
parallel FBP comes back flipped. With the image and detector on [-1, 1] the
forward projectors agree with torchtomo's with no fitted scale.

ASTRA is an optional dependency: pip install astra-toolbox.
"""

import astra
import astra.experimental
import numpy as np
import torch
from matched_pair import MatchedPairMixin

from torchtomo import FanBeam, ParallelBeam

# torchtomo's image spans [-1, 1] on both axes.
VOLUME_WINDOW = (-1.0, 1.0, -1.0, 1.0)


class _Fbp:
    """ASTRA's FBP_CUDA on one sinogram at a time, through buffers linked once."""

    def __init__(self, projection_geometry, size, n_angles, n_det, device):
        self.sinogram = torch.zeros(n_angles, n_det, device=device)
        self.image = torch.zeros(size, size, device=device)
        self.sinogram_id = astra.data2d.link("-sino", projection_geometry, self.sinogram)
        self.image_id = astra.data2d.link("-vol", astra.create_vol_geom(size, size, *VOLUME_WINDOW), self.image)
        config = astra.astra_dict("FBP_CUDA")
        config["ProjectionDataId"] = self.sinogram_id
        config["ReconstructionDataId"] = self.image_id
        config["option"] = {"FilterType": "ram-lak"}
        self.algorithm_id = astra.algorithm.create(config)

    def __call__(self, sinogram):
        self.sinogram.copy_(sinogram)
        astra.algorithm.run(self.algorithm_id)
        return self.image.clone()

    def __del__(self):
        astra.algorithm.delete(self.algorithm_id)
        astra.data2d.delete([self.sinogram_id, self.image_id])


class _AstraBase(MatchedPairMixin):
    def compute_device(self, tensor):
        if tensor.is_cuda:
            return tensor.device
        if not torch.cuda.is_available():
            raise RuntimeError("the ASTRA adapter here runs on CUDA only")
        return torch.device("cuda", 0)

    def astra_angles(self):
        return -self.angles.detach().cpu().numpy().astype(np.float64)

    def fbp_operator(self, device):
        if getattr(self, "_fbp", None) is None or self._fbp.image.device != device:
            self._fbp = _Fbp(self.fbp_geometry(), self.img_size, self.n_angles, self.n_det, device)
        return self._fbp

    @torch.no_grad()
    def fbp(self, sinogram, filter_name="ramp"):
        """ASTRA's own filtered backprojection, with its Ram-Lak filter."""
        if filter_name != "ramp":
            raise NotImplementedError(f"the ASTRA adapter offers the ramp filter only, not {filter_name!r}")
        device = self.compute_device(sinogram)
        operator = self.fbp_operator(device)
        images = [self.from_astra_2d(operator(view.to(device))) for view in sinogram[:, 0]]
        return self.masked(torch.stack(images).unsqueeze(1).to(sinogram.device))


class AstraParallelBeam(_AstraBase, ParallelBeam):
    """ParallelBeam with ASTRA's CUDA kernels behind forward, adjoint, and FBP."""

    def __init__(
        self, img_size=256, n_angles=180, n_det=None, angle_range=(0, np.pi), circle=True, angles=None, **kwargs
    ):
        super().__init__(img_size, n_angles, n_det, angle_range, circle, angles=angles, **kwargs)
        if self.n_det != img_size:
            raise ValueError("the ASTRA geometry here assumes a detector as wide as the image")
        self._projectors = {}
        self._fbp = None

    def astra_projector(self, slices):
        """One 3D projector per batch size: ASTRA holds the slice count in the geometry."""
        if slices not in self._projectors:
            pixel = 2.0 / self.img_size
            volume = astra.create_vol_geom(
                self.img_size, self.img_size, slices, *VOLUME_WINDOW, -slices * pixel / 2, slices * pixel / 2
            )
            projection = astra.create_proj_geom("parallel3d", pixel, pixel, slices, self.n_det, self.astra_angles())
            self._projectors[slices] = astra.create_projector("cuda3d", projection, volume)
        return self._projectors[slices]

    def fbp_geometry(self):
        return astra.create_proj_geom("parallel", 2.0 / self.img_size, self.n_det, self.astra_angles())

    def from_astra_2d(self, image):
        return image.flip(-2)

    @torch.no_grad()
    def project_raw(self, image):
        """[B, 1, H, W] -> [B, 1, angles, detectors]; ASTRA's 3D layout is [rows, angles, columns]."""
        device = self.compute_device(image)
        volume = image[:, 0].detach().to(device).contiguous()
        sinogram = torch.zeros(volume.shape[0], self.n_angles, self.n_det, device=device)
        astra.experimental.direct_FP3D(self.astra_projector(volume.shape[0]), volume, sinogram)
        return sinogram.unsqueeze(1).to(image.device)

    @torch.no_grad()
    def backproject_raw(self, sinogram):
        device = self.compute_device(sinogram)
        views = sinogram[:, 0].detach().to(device).contiguous()
        volume = torch.zeros(views.shape[0], self.img_size, self.img_size, device=device)
        astra.experimental.direct_BP3D(self.astra_projector(views.shape[0]), volume, views)
        return volume.unsqueeze(1).to(sinogram.device)

    def __del__(self):
        for projector in self._projectors.values():
            astra.projector3d.delete(projector)


class AstraFanBeam(_AstraBase, FanBeam):
    """FanBeam with ASTRA's 2D fan-beam CUDA kernels, one image at a time."""

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
        volume = astra.create_vol_geom(img_size, img_size, *VOLUME_WINDOW)
        self._projector = astra.create_projector("cuda", self.fbp_geometry(), volume)
        self._fbp = None

    def fbp_geometry(self):
        pixel = 2.0 / self.img_size
        return astra.create_proj_geom(
            "fanflat",
            self.det_width / self.n_det * pixel,
            self.n_det,
            self.astra_angles(),
            self.src_dist * pixel,
            self.det_dist * pixel,
        )

    def from_astra_2d(self, image):
        return image

    @torch.no_grad()
    def project_raw(self, image):
        device = self.compute_device(image)
        images = image[:, 0].detach().to(device).contiguous()
        sinogram = torch.zeros(images.shape[0], self.n_angles, self.n_det, device=device)
        for index in range(images.shape[0]):
            astra.experimental.direct_FP2D(self._projector, images[index], sinogram[index])
        return sinogram.unsqueeze(1).to(image.device)

    @torch.no_grad()
    def backproject_raw(self, sinogram):
        device = self.compute_device(sinogram)
        views = sinogram[:, 0].detach().to(device).contiguous()
        volume = torch.zeros(views.shape[0], self.img_size, self.img_size, device=device)
        for index in range(views.shape[0]):
            astra.experimental.direct_BP2D(self._projector, volume[index], views[index])
        return volume.unsqueeze(1).to(sinogram.device)

    def __del__(self):
        astra.projector.delete(self._projector)
