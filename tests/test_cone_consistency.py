"""torchtomo's 3D cone beam and breast tomosynthesis against ASTRA and LEAP (CUDA required).

Each library reads torchtomo's source and detector positions through
``cone_projectors.py``. A circular scan with an off-centre, anisotropic volume
and a tomosynthesis arc with thick slices must give the same projections and
backprojections with no fitted scale, and each library's pair must pass a
dot-product test. A volume flipped in x or y catches a transposed convention.

ConeBeam and BreastTomosynthesis are newer than torchtomo 0.4.0; these tests skip
when the installed torchtomo lacks them.

    pytest tests/test_cone_consistency.py
"""

import math

import pytest
import torch

torchtomo = pytest.importorskip("torchtomo")
if not hasattr(torchtomo, "ConeBeam") or not hasattr(torchtomo, "BreastTomosynthesis"):
    pytest.skip("this torchtomo has no ConeBeam or BreastTomosynthesis", allow_module_level=True)

from compare_cone import Case, circular_scan, dot_test, operator, relative_l2, shepp_logan_3d  # noqa: E402

try:
    from cone_projectors import AstraConeBeam
except ImportError:
    AstraConeBeam = None

try:
    from cone_projectors import tomographicModels
except ImportError:
    tomographicModels = None

LIBRARIES = [
    pytest.param("astra", AstraConeBeam is not None, 0.03, 0.08, id="astra"),
    pytest.param("leap", tomographicModels is not None, 0.04, 0.15, id="leap"),
]


def _cone_case():
    scan = circular_scan(32, 36, "cuda")
    return Case(
        "off-centre cone",
        (24, 32, 40),
        scan.sources,
        scan.detectors,
        (1.5, 1.0, 1.0),
        (3.0, -2.0, 4.0),
        96,
    )


def _dbt_case():
    tomosynthesis = torchtomo.BreastTomosynthesis(
        (16, 48, 96),
        n_angles=9,
        detector_shape=(128, 64),
        voxel_spacing=(2.0, 1.0, 1.0),
        volume_center=(0.0, 0.0, -30.0),
        device="cuda",
    )
    return Case(
        "tomosynthesis",
        (16, 48, 96),
        tomosynthesis.source_positions.detach(),
        tomosynthesis.detector_positions.detach(),
        (2.0, 1.0, 1.0),
        (0.0, 0.0, -30.0),
        math.ceil(1.3 * 32),
    )


def _volume(case):
    """A Shepp-Logan phantom under a ramp along every axis, so no flip leaves it unchanged."""
    cube = shepp_logan_3d(max(case.volume_shape), "cuda")
    resized = torch.nn.functional.interpolate(cube, size=case.volume_shape, mode="trilinear")
    ramps = [torch.linspace(0.5, 1.5, size, device="cuda") for size in case.volume_shape]
    z, y, x = torch.meshgrid(*ramps, indexing="ij")
    return resized * z * y.square() * x.pow(3)


CASES = [pytest.param(_cone_case, id="cone"), pytest.param(_dbt_case, id="dbt")]


def _pair(library, available, make_case):
    if not available or not torch.cuda.is_available():
        pytest.skip(f"{library} and CUDA are required")
    case = make_case()
    return case, operator("torchtomo", case), operator(library, case)


@pytest.mark.parametrize("make_case", CASES)
@pytest.mark.parametrize("library, available, forward_tolerance, adjoint_tolerance", LIBRARIES)
def test_forward_matches_torchtomo(library, available, forward_tolerance, adjoint_tolerance, make_case):
    case, ours, theirs = _pair(library, available, make_case)
    volume = _volume(case)
    mine = ours(volume)
    assert relative_l2(mine, theirs.forward(volume)) < forward_tolerance
    for axis in (-1, -2):
        assert relative_l2(mine, theirs.forward(volume.flip(axis))) > 0.3


@pytest.mark.parametrize("make_case", CASES)
@pytest.mark.parametrize("library, available, forward_tolerance, adjoint_tolerance", LIBRARIES)
def test_adjoint_matches_torchtomo(library, available, forward_tolerance, adjoint_tolerance, make_case):
    case, ours, theirs = _pair(library, available, make_case)
    projections = ours(_volume(case))
    assert relative_l2(ours.adjoint(projections), theirs.adjoint(projections)) < adjoint_tolerance


@pytest.mark.parametrize("make_case", CASES)
@pytest.mark.parametrize("library, available, forward_tolerance, adjoint_tolerance", LIBRARIES)
def test_dot_product(library, available, forward_tolerance, adjoint_tolerance, make_case):
    _, ours, theirs = _pair(library, available, make_case)
    generator = torch.Generator(device="cuda").manual_seed(0)
    assert abs(dot_test(ours, generator) - 1) < 1e-4
    assert abs(dot_test(theirs, generator) - 1) < 0.1
