"""ASTRA and LEAP cone beam operators driven by torchtomo's explicit geometry.

torchtomo's ConeBeam describes a scan by its source positions ``[views, 3]`` and
the centre of every detector element ``[views, rows, columns, 3]``, in ``(x, y, z)``
order, with volumes ``[B, 1, Z, Y, X]`` and projections ``[B, 1, views, rows,
columns]``. Both adapters read the same tensors, so a circular scan and a breast
tomosynthesis arc reach all three libraries as one geometry:

- ASTRA's ``cone_vec`` takes, per view, the source, the panel centre, and the
  steps from one detector element to the next along a row (``u``) and down a
  column (``v``). Its projections are ``[rows, views, columns]``.
- LEAP's modular beam takes the same source and panel centre with unit row and
  column vectors and the element pitch. Its projections are ``[views, rows,
  columns]`` and its volume voxels are square in ``x`` and ``y``.

Both work on the torch tensors in place on the card. Panels must be flat with a
uniform pitch, which is what ``cone_vec`` and modular beams describe.
"""

import ctypes
import math
from pathlib import Path

import astra
import astra.experimental
import numpy as np
import torch

try:
    from leapctype import tomographicModels
except ImportError:
    tomographicModels = None


def load_leap_runtime():
    """Make cudaGetDeviceProperties_v2 resolvable for a LEAP built against CUDA 12.

    LEAP links its CUDA runtime statically except for that one symbol. A CUDA 13
    PyTorch loads only libcudart.so.13, which no longer exports it, so the pip
    CUDA 12 runtime is loaded globally first when it is installed.
    """
    try:
        import nvidia.cuda_runtime
    except ImportError:
        return
    for directory in nvidia.cuda_runtime.__path__:
        for library in sorted(Path(directory, "lib").glob("libcudart.so.12*")):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            return


def flat_panel(detector_positions):
    """Panel centres and the element steps along a row (u) and down a column (v), per view."""
    detectors = detector_positions.detach().double()
    rows, columns = detectors.shape[1:3]
    corner = detectors[:, 0, 0]
    u = (detectors[:, 0, -1] - corner) / max(columns - 1, 1)
    v = (detectors[:, -1, 0] - corner) / max(rows - 1, 1)
    row_index = torch.arange(rows, dtype=detectors.dtype, device=detectors.device)
    column_index = torch.arange(columns, dtype=detectors.dtype, device=detectors.device)
    ideal = (
        corner[:, None, None] + row_index[:, None, None] * v[:, None, None] + column_index[:, None] * u[:, None, None]
    )
    deviation = float((ideal - detectors).norm(dim=-1).amax() / u.norm(dim=-1).amin())
    if deviation > 1e-3:
        raise ValueError(f"the panel is not flat with a uniform pitch: off by {deviation:.2g} elements")
    centres = corner + (columns - 1) / 2 * u + (rows - 1) / 2 * v
    return centres, u, v


def volume_bounds(volume_shape, voxel_spacing, volume_center):
    """(x0, x1, y0, y1, z0, z1) of the volume's outer faces; spacing is (z, y, x)."""
    spacing = np.broadcast_to(np.asarray(voxel_spacing, dtype=np.float64), (3,))
    half = np.asarray(volume_shape, dtype=np.float64)[::-1] * spacing[::-1] / 2
    center = np.asarray(volume_center, dtype=np.float64)
    return tuple(value for axis in range(3) for value in (center[axis] - half[axis], center[axis] + half[axis]))


class AstraConeBeam:
    """ASTRA's cuda3d projector and backprojector on a cone_vec geometry, one volume at a time."""

    name = "astra"

    def __init__(self, volume_shape, source_positions, detector_positions, voxel_spacing=1.0, volume_center=(0, 0, 0)):
        self.volume_shape = tuple(volume_shape)
        self.projection_shape = tuple(detector_positions.shape[:3])
        centres, u, v = flat_panel(detector_positions)
        vectors = torch.cat([source_positions.detach().double(), centres, u, v], dim=-1).cpu().numpy()
        views, rows, columns = self.projection_shape
        depth, height, width = self.volume_shape
        self.volume_geometry = astra.create_vol_geom(
            height, width, depth, *volume_bounds(volume_shape, voxel_spacing, volume_center)
        )
        self.projection_geometry = astra.create_proj_geom("cone_vec", rows, columns, vectors)
        self.projector = astra.create_projector("cuda3d", self.projection_geometry, self.volume_geometry)

    @torch.no_grad()
    def forward(self, volume):
        projections = []
        for item in volume[:, 0]:
            views, rows, columns = self.projection_shape
            output = torch.zeros(rows, views, columns, device=volume.device, dtype=torch.float32)
            astra.experimental.direct_FP3D(self.projector, item.float().contiguous(), output)
            projections.append(output.permute(1, 0, 2))
        return torch.stack(projections)[:, None].to(volume.dtype)

    @torch.no_grad()
    def adjoint(self, projections):
        volumes = []
        for item in projections[:, 0]:
            output = torch.zeros(self.volume_shape, device=projections.device, dtype=torch.float32)
            astra.experimental.direct_BP3D(self.projector, output, item.float().permute(1, 0, 2).contiguous())
            volumes.append(output)
        return torch.stack(volumes)[:, None].to(projections.dtype)

    def fdk(self, projections):
        """ASTRA's FDK_CUDA, which supports circular cone_vec scans, on one scan at a time."""
        volumes = []
        for item in projections[:, 0]:
            output = torch.zeros(self.volume_shape, device=projections.device, dtype=torch.float32)
            data = item.float().permute(1, 0, 2).contiguous()
            projection_id = astra.data3d.link("-sino", self.projection_geometry, data)
            volume_id = astra.data3d.link("-vol", self.volume_geometry, output)
            config = astra.astra_dict("FDK_CUDA")
            config["ProjectionDataId"] = projection_id
            config["ReconstructionDataId"] = volume_id
            algorithm = astra.algorithm.create(config)
            try:
                astra.algorithm.run(algorithm)
            finally:
                astra.algorithm.delete(algorithm)
                astra.data3d.delete([projection_id, volume_id])
            volumes.append(output)
        return torch.stack(volumes)[:, None].to(projections.dtype)

    def __del__(self):
        astra.projector3d.delete(self.projector)


def slice_interpolation(depth, slice_spacing, fine_spacing):
    """Linear interpolation from ``depth`` slices to slices ``fine_spacing`` thick, over the same box.

    Returns ``[fine, depth]``; fine slices whose centres fall outside the box are zero,
    and those between the outermost slice centres and the faces take the nearest slice.
    """
    thickness = depth * slice_spacing
    fine = math.ceil(thickness / fine_spacing - 1e-6)
    centres = (np.arange(fine) - (fine - 1) / 2) * fine_spacing
    position = np.clip(centres / slice_spacing + (depth - 1) / 2, 0, depth - 1)
    lower = np.minimum(np.floor(position).astype(int), depth - 2)
    weights = np.zeros((fine, depth))
    rows = np.arange(fine)
    weights[rows, lower] = lower + 1 - position
    weights[rows, lower + 1] = position - lower
    weights[np.abs(centres) > thickness / 2] = 0
    return weights


class LeapConeBeam:
    """LEAP's modular-beam projector and its matched backprojector, one volume at a time.

    LEAP's modular beam needs cubic voxels. Thicker slices, as in tomosynthesis, are
    interpolated linearly onto cubic voxels before projection, and the
    backprojection is mapped back with the transpose, so the pair stays matched.
    """

    name = "leap"

    def __init__(self, volume_shape, source_positions, detector_positions, voxel_spacing=1.0, volume_center=(0, 0, 0)):
        if tomographicModels is None:
            raise ImportError("LEAP is not installed")
        load_leap_runtime()
        self.volume_shape = tuple(volume_shape)
        self.projection_shape = tuple(detector_positions.shape[:3])
        centres, u, v = flat_panel(detector_positions)
        spacing = np.broadcast_to(np.asarray(voxel_spacing, dtype=np.float64), (3,))
        if not np.isclose(spacing[1], spacing[2]):
            raise ValueError("LEAP voxels are square in x and y")
        views, rows, columns = self.projection_shape
        depth, height, width = self.volume_shape
        self.interpolation = None
        if not np.isclose(spacing[0], spacing[1]):
            weights = slice_interpolation(depth, spacing[0], spacing[1])
            self.interpolation = torch.as_tensor(weights, dtype=torch.float32, device=source_positions.device)
            depth = weights.shape[0]
        self.leap_shape = (depth, height, width)
        model = tomographicModels()
        model.print_warnings = False
        model.set_gpu(torch.cuda.current_device())
        model.set_modularBeam(
            views,
            rows,
            columns,
            float(v.norm(dim=-1).mean()),
            float(u.norm(dim=-1).mean()),
            *(
                np.ascontiguousarray(array.cpu().numpy(), dtype=np.float32)
                for array in (
                    source_positions.detach().double(),
                    centres,
                    v / v.norm(dim=-1, keepdim=True),
                    u / u.norm(dim=-1, keepdim=True),
                )
            ),
        )
        model.set_volume(width, height, depth, float(spacing[2]), float(spacing[1]), *map(float, volume_center))
        self.model = model

    @torch.no_grad()
    def forward(self, volume):
        projections = []
        for item in volume[:, 0].float():
            if self.interpolation is not None:
                item = torch.einsum("fz,zyx->fyx", self.interpolation, item)
            output = torch.zeros(self.projection_shape, device=volume.device, dtype=torch.float32)
            self.model.project(output, item.contiguous())
            projections.append(output)
        return torch.stack(projections)[:, None].to(volume.dtype)

    @torch.no_grad()
    def adjoint(self, projections):
        volumes = []
        for item in projections[:, 0]:
            output = torch.zeros(self.leap_shape, device=projections.device, dtype=torch.float32)
            self.model.backproject(item.float().contiguous().clone(), output)
            if self.interpolation is not None:
                output = torch.einsum("fz,fyx->zyx", self.interpolation, output)
            volumes.append(output)
        return torch.stack(volumes)[:, None].to(projections.dtype)
