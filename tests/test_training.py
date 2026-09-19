"""The training and comparison code keeps the geometry torchtomo was asked for."""

import argparse

import torch

from torchtomo import FanBeam, ParallelBeam


def test_subset_keeps_fan_geometry():
    from classical import subset_projectors

    parent = FanBeam(img_size=32, n_angles=12, n_samples=8)
    pairs = subset_projectors(parent, 3)
    assert len(pairs) == 3
    for indices, subset in pairs:
        assert subset.src_dist == parent.src_dist
        assert subset.det_dist == parent.det_dist
        assert subset.n_det == parent.n_det
        assert subset.n_samples == parent.n_samples
        torch.testing.assert_close(subset.angles, parent.angles[indices])


def test_subset_parallel_does_not_grow_fan_kwargs():
    from classical import subset_projectors

    parent = ParallelBeam(img_size=32, n_angles=12)
    _, subset = subset_projectors(parent, 3)[0]
    assert not hasattr(subset, "src_dist")
    assert subset.n_det == parent.n_det


def _assert_iradonmap_matches_fbp(projector):
    from models import IRadonMap

    model = IRadonMap(projector, width=4)
    y = torch.randn(2, 1, projector.n_angles, projector.n_det)
    with torch.no_grad():
        learned = model.sinusoidal_backprojection(y @ model.filtering.t())
        analytic = projector.fbp(y)
    torch.testing.assert_close(learned, analytic, rtol=1e-4, atol=1e-5)


def test_iradonmap_untrained_matches_parallel_fbp():
    _assert_iradonmap_matches_fbp(ParallelBeam(img_size=32, n_angles=16))


def test_iradonmap_untrained_matches_fan_fbp():
    _assert_iradonmap_matches_fbp(FanBeam(img_size=32, n_angles=16, n_samples=8))


def test_compare_libraries_parallel_default_and_fan_span():
    from compare_libraries import angle_tensor, available_backends

    assert available_backends()[0].__name__ == "TorchtomoBackend"
    assert available_backends("fan")[0].__name__ == "TorchtomoFanBackend"
    parallel = angle_tensor(10, torch.device("cpu"), "parallel")
    fan = angle_tensor(10, torch.device("cpu"), "fan")
    torch.testing.assert_close(parallel[-1], torch.tensor(9 * torch.pi / 10))
    torch.testing.assert_close(fan[-1], torch.tensor(9 * 2 * torch.pi / 10))


def test_train_build_projector_default_stays_parallel():
    from train import build_projector

    parallel_args = argparse.Namespace(
        geometry="parallel",
        projector="torchtomo",
        image_size=32,
        angles=16,
        src_dist=None,
        det_dist=None,
        n_det=None,
    )
    assert isinstance(build_projector(parallel_args), ParallelBeam)
    fan_args = argparse.Namespace(**{**parallel_args.__dict__, "geometry": "fan"})
    fan = build_projector(fan_args)
    assert isinstance(fan, FanBeam)
    assert fan.src_dist == 64
    assert fan.n_det == 48
