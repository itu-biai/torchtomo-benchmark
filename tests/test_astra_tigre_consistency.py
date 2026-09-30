"""torchtomo against ASTRA and TIGRE, through the drop-in adapters (CUDA required).

Both are independent CUDA implementations. Like LEAP, each one agrees with
torchtomo's forward projector and adjoint with no fitted scale once its angle and
detector conventions are mapped, and its own FBP reconstructs as well as
torchtomo's.

    pytest tests/test_astra_tigre_consistency.py
"""

import pytest
import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan

try:
    from astra_projector import AstraFanBeam, AstraParallelBeam
except ImportError:
    AstraParallelBeam = AstraFanBeam = None

try:
    from tigre_projector import TigreFanBeam, TigreParallelBeam
except ImportError:
    TigreParallelBeam = TigreFanBeam = None

ADAPTERS = [
    pytest.param(AstraParallelBeam, ParallelBeam, 0.01, id="astra-parallel"),
    pytest.param(AstraFanBeam, FanBeam, 0.02, id="astra-fan"),
    pytest.param(TigreParallelBeam, ParallelBeam, 0.01, id="tigre-parallel"),
    pytest.param(TigreFanBeam, FanBeam, 0.02, id="tigre-fan"),
]


def _relative_l2(reference, other):
    return float((reference - other).norm() / reference.norm())


def _psnr(truth, image, mask):
    error = float((((truth - image) * mask) ** 2).sum() / mask.sum())
    return 10.0 * torch.log10(torch.tensor(1.0 / error)).item()


def _pair(adapter, reference_class, size=256, n_angles=180):
    if adapter is None or not torch.cuda.is_available():
        pytest.skip("the library and CUDA are required")
    ours = reference_class(img_size=size, n_angles=n_angles).to("cuda")
    theirs = adapter(img_size=size, n_angles=n_angles, angles=ours.angles.cpu()).to("cuda")
    truth = shepp_logan(size).reshape(1, 1, size, size).cuda() * ours.circle_mask
    return ours, theirs, torch.cat([truth, truth.flip(-1).transpose(-1, -2)])


@pytest.mark.parametrize("adapter, reference_class, tolerance", ADAPTERS)
def test_forward_matches_torchtomo(adapter, reference_class, tolerance):
    """Same image, same sinogram, no fitted scale; the second image catches a transposed axis."""
    ours, theirs, truth = _pair(adapter, reference_class)
    assert _relative_l2(ours.forward(truth), theirs.forward(truth)) < tolerance


@pytest.mark.parametrize("adapter, reference_class, tolerance", ADAPTERS)
def test_adjoint_matches_torchtomo(adapter, reference_class, tolerance):
    ours, theirs, truth = _pair(adapter, reference_class)
    sinogram = ours.forward(truth)
    adjoint, other = ours.adjoint(sinogram), theirs.adjoint(sinogram)
    assert _relative_l2(adjoint, other) < 2 * tolerance
    scale = float(other.abs().mean() / adjoint.abs().mean())
    assert 0.98 < scale < 1.02, f"backprojection scale {scale:.4f} is not unity"


@pytest.mark.parametrize("adapter, reference_class, tolerance", ADAPTERS)
def test_fbp_quality_matches_torchtomo(adapter, reference_class, tolerance):
    """Each library's own FBP of its own sinogram lands within a decibel of torchtomo's."""
    ours, theirs, truth = _pair(adapter, reference_class)
    mask = ours.circle_mask
    mine = _psnr(truth, ours.fbp(ours.forward(truth)), mask)
    other = _psnr(truth, theirs.fbp(theirs.forward(truth)), mask)
    assert abs(mine - other) < 1.0, f"FBP PSNR gap: torchtomo {mine:.2f} dB, {adapter.__name__} {other:.2f} dB"


@pytest.mark.parametrize("adapter, reference_class, tolerance", ADAPTERS)
def test_gradient_is_the_libraries_backprojection(adapter, reference_class, tolerance):
    ours, theirs, truth = _pair(adapter, reference_class)
    image = truth.clone().requires_grad_(True)
    sinogram = theirs.forward(image)
    weights = torch.rand_like(sinogram)
    (sinogram * weights).sum().backward()
    assert torch.allclose(image.grad, theirs.adjoint(weights), rtol=1e-4, atol=1e-4)
