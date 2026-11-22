"""Tests for model registry safety rules and feathered tiled inference."""

from pathlib import Path

import pytest
import torch

from ruby_ai.models import MODEL_REGISTRY, infer_tiled, resolve_model_path


def test_registry_uses_safetensors_and_1x_cc_by() -> None:
    assert {"de-h264", "denoise", "de-jpg"}.issubset(MODEL_REGISTRY.keys())
    for spec in MODEL_REGISTRY.values():
        assert spec.scale == 1
        assert spec.filename.endswith(".safetensors")
        assert spec.license == "CC-BY-4.0"


def test_pickle_checkpoint_rejected_without_opt_in(tmp_path: Path) -> None:
    fake_pth = tmp_path / "model.pth"
    fake_pth.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="allow-unsafe-pickle"):
        resolve_model_path(str(fake_pth), allow_unsafe_pickle=False)

    resolved, spec = resolve_model_path(str(fake_pth), allow_unsafe_pickle=True)
    assert resolved == fake_pth
    assert spec is None


def test_infer_tiled_identity_and_constant_shift_on_non_divisible_shape() -> None:
    torch.manual_seed(7)
    image = torch.rand((1, 3, 130, 190), dtype=torch.float32)

    def identity(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0
        return x

    def shift(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] % 4 == 0 and x.shape[3] % 4 == 0
        return x + 0.125

    out_id = infer_tiled(
        identity,
        image,
        tile_size=64,
        tile_overlap=16,
        multiple_of=4,
    )
    assert out_id.shape == image.shape
    assert (out_id - image).abs().max().item() < 1e-5

    out_shift = infer_tiled(
        shift,
        image,
        tile_size=64,
        tile_overlap=16,
        multiple_of=4,
    )
    assert out_shift.shape == image.shape
    assert (out_shift - (image + 0.125)).abs().max().item() < 1e-5


def test_infer_tiled_full_frame_padding() -> None:
    image = torch.rand((1, 3, 35, 51), dtype=torch.float32)

    def double_identity(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] >= 64 and x.shape[3] >= 64
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0
        return x

    out = infer_tiled(
        double_identity,
        image,
        tile_size=0,
        tile_overlap=0,
        multiple_of=8,
        minimum=64,
    )
    assert out.shape == image.shape
    assert (out - image).abs().max().item() < 1e-6
