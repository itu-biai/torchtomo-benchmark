"""Regressions for masking independence, honest metrics and validation selection."""

import argparse

import pytest
import torch
from compare_libraries import compatible_results, ssim
from objectives import perturb_projections

from torchtomo import ParallelBeam


@pytest.mark.parametrize("grid", [2, 3, 4])
def test_masked_measurements_cannot_affect_perturbed_input(grid):
    original = torch.randn(2, 1, 12, 16)
    for index in range(grid * grid):
        perturbed, mask = perturb_projections(original, index, grid)
        changed = original.clone()
        changed[mask] += 1000
        other, _ = perturb_projections(changed, index, grid)
        torch.testing.assert_close(perturbed, other)
        torch.testing.assert_close(perturbed[~mask], original[~mask])


def test_ssim_excludes_perfect_invisible_corners():
    truth = torch.zeros(1, 1, 32, 32)
    image = truth.clone()
    mask = torch.zeros_like(truth)
    mask[..., 8:24, 8:24] = 1
    truth[mask.bool()] = 1
    # Square SSIM is flattered by matching zero corners. Circle SSIM is poor.
    from skimage.metrics import structural_similarity

    square = structural_similarity(truth.squeeze().numpy(), image.squeeze().numpy(), data_range=1)
    assert ssim(truth, image, mask) < square - 0.1


def test_partial_results_cannot_mix_environments():
    current = {
        "device": "gpu",
        "torch": "2",
        "geometry": "parallel",
        "libraries": ["torchtomo"],
        "metadata": {"torchtomo_version": "0.4.0"},
    }
    old = {**current, "metadata": {"torchtomo_version": "0.3.0"}, "performance": [1]}
    with pytest.raises(ValueError, match="metadata differs"):
        compatible_results(current, old, ["quality"])
    assert compatible_results(current, old, ["quality", "performance"]) == {}
    assert compatible_results(current, current, ["quality"]) == current


def test_tuned_fbp_selects_only_on_validation():
    from classical import run_classical

    projector = ParallelBeam(img_size=16, n_angles=8, backend="torch")
    noisy = torch.randn(2, 1, 8, 16)
    # Validation is exactly Hann; test is exactly ramp. Selection must stay Hann.
    truth = torch.cat([projector.fbp(noisy[:1], "hann"), projector.fbp(noisy[1:], "ramp")])
    data = {
        "noisy": noisy,
        "truth": truth,
        "fbp": projector.fbp(noisy),
        "splits": {"val": torch.tensor([0]), "test": torch.tensor([1])},
    }
    args = argparse.Namespace(
        device="cpu",
        batch_size=1,
        red_iterations=1,
        sirt_iterations=1,
        sirt_relaxation=1,
        sart_sweeps=1,
        sart_subsets=2,
        sart_relaxations="1",
        bm3d_sigmas="0.1",
    )
    reconstructions, selection = run_classical(projector, data, None, 1, args, projector.circle_mask.bool())
    assert selection["fbp-tuned"]["selected_setting"] == "hann"
    torch.testing.assert_close(reconstructions["fbp-tuned"], projector.fbp(noisy[1:], "hann"))


def test_repeat_summary_uses_training_runs_and_rejects_dose_changes(tmp_path):
    import json

    from summarize_repeats import summarize_runs

    paths = []
    for seed, score in [(1, 20), (2, 22)]:
        directory = tmp_path / str(seed)
        directory.mkdir()
        config = {
            "methodology_revision": 2,
            "dose_protocol": "fixed",
            "truth_sha256": "same-data",
            "photons": 100000,
            "seed": 2026,
            "training_seed": seed,
            "projector": "torchtomo",
            "backend": "cuda",
        }
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "metrics.json").write_text(
            json.dumps({"ids": [3, 4], "methods": {"lpd": {"mean_roi_psnr_db": score}}})
        )
        paths.append(directory)
    row = summarize_runs(paths)["backends"]["torchtomo:cuda"]["methods"]["lpd"]
    assert row["mean_db"] == 21
    assert row["std_across_seeds_db"] == pytest.approx(2**0.5)
    config["photons"] = 200000
    (paths[1] / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="photons"):
        summarize_runs(paths)


def test_torchradon_fan_uses_the_same_scan():
    from compare_libraries import RadonFanbeam, TorchRadonFanBackend, angle_tensor

    from torchtomo import FanBeam, shepp_logan

    if RadonFanbeam is None or not torch.cuda.is_available():
        pytest.skip("torch-radon and CUDA required")
    size = 256
    angles = angle_tensor(24, torch.device("cuda"), "fan")
    projector = FanBeam(img_size=size, n_angles=24, angles=angles, backend="cuda").cuda()
    image = shepp_logan(size).cuda()
    reference = projector(image)
    other = TorchRadonFanBackend(size, angles, torch.device("cuda")).forward(image)
    # Physical pixel-width scale, not a fitted gain. The asymmetric phantom catches arc reversal.
    residual = (reference - other * (2 / size)).norm() / reference.norm()
    assert residual < 0.015


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
def test_leap_filter_search_does_not_change_the_ramp(geometry):
    from compare_libraries import LeapFanBeam, LeapParallelBeam

    cls = LeapParallelBeam if geometry == "parallel" else LeapFanBeam
    if cls is None or not torch.cuda.is_available():
        pytest.skip("LEAP and CUDA required")
    projector = cls(img_size=32, n_angles=12).cuda()
    sinogram = torch.randn(1, 1, 12, projector.n_det, device="cuda")
    original = projector.fbp(sinogram)
    smooth = projector.fbp(sinogram, "leap-order-0")
    assert not torch.allclose(original, smooth)
    torch.testing.assert_close(projector.fbp(sinogram), original)
