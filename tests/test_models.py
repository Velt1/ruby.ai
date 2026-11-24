"""Tests for model registry extensibility, FastGroupNorm, auto-tiling, and batched inference."""

import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from ruby_ai.device import DeviceInfo, resolve_auto_tile_size
from ruby_ai.models import (
    ALLOWED_MODEL_LICENSES,
    MODEL_REGISTRY,
    DirectUrlSource,
    FastGroupNorm,
    ModelRegistry,
    ModelSpec,
    infer_tiled,
    resolve_model_path,
)


def test_registry_uses_safetensors_and_1x_verified_licenses() -> None:
    expected_keys = {
        "de-h264",
        "denoise",
        "de-jpg",
        "superscale-span",
        "superscale-fast",
        "de-jpg-fast",
        "exposure-compact",
    }
    assert expected_keys.issubset(set(MODEL_REGISTRY.keys()))

    for spec in MODEL_REGISTRY.values():
        assert spec.scale == 1
        assert spec.filename.endswith(".safetensors")
        assert spec.license in ALLOWED_MODEL_LICENSES

    for core_key in ("de-h264", "denoise", "de-jpg"):
        assert MODEL_REGISTRY[core_key].license == "CC-BY-4.0"
        assert MODEL_REGISTRY[core_key].commercial_use is True


def test_model_registry_extensible_register() -> None:
    registry = ModelRegistry(MODEL_REGISTRY.values())
    custom = ModelSpec(
        key="custom-test",
        source=DirectUrlSource(
            url="https://example.invalid/custom.safetensors",
            filename="custom.safetensors",
        ),
        architecture="SPAN",
        scale=1,
        author="Tester",
        license="MIT",
        description="Custom test model",
        source_url="https://example.invalid",
    )
    registry.register(custom)
    assert "custom-test" in registry
    assert registry["custom-test"] is custom
    with pytest.raises(ValueError, match="already registered"):
        registry.register(custom)


def test_direct_url_source_caching_and_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUBY_AI_CACHE_DIR", str(tmp_path))
    payload = b"dummy safetensors header and payload"
    digest = hashlib.sha256(payload).hexdigest()

    class FakeResp(io.BytesIO):
        def __enter__(self) -> "FakeResp":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    calls = 0

    def fake_urlopen(req: object, timeout: float = 60) -> FakeResp:
        nonlocal calls
        del req, timeout
        calls += 1
        return FakeResp(payload)

    src = DirectUrlSource(
        url="https://example.invalid/model.safetensors",
        filename="test_model.safetensors",
        sha256=digest,
    )
    with patch("ruby_ai.models.urllib.request.urlopen", side_effect=fake_urlopen):
        first = src.resolve()
        second = src.resolve()

    assert first == second
    assert first.read_bytes() == payload
    assert calls == 1

    bad_src = DirectUrlSource(
        url="https://example.invalid/bad.safetensors",
        filename="bad_model.safetensors",
        sha256="0" * 64,
    )
    with (
        patch("ruby_ai.models.urllib.request.urlopen", side_effect=fake_urlopen),
        pytest.raises(ValueError, match="SHA-256 mismatch"),
    ):
        bad_src.resolve()
    assert not (tmp_path / "bad_model.safetensors").exists()


def test_pickle_checkpoint_rejected_without_opt_in(tmp_path: Path) -> None:
    fake_pth = tmp_path / "model.pth"
    fake_pth.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="allow-unsafe-pickle"):
        resolve_model_path(str(fake_pth), allow_unsafe_pickle=False)

    resolved, spec = resolve_model_path(str(fake_pth), allow_unsafe_pickle=True)
    assert resolved == fake_pth
    assert spec is None


def test_fast_group_norm_matches_nn_group_norm() -> None:
    torch.manual_seed(42)
    ref = nn.GroupNorm(num_groups=4, num_channels=64, eps=1e-5)
    fast = FastGroupNorm(ref)

    x_nchw = torch.randn((2, 64, 24, 32), dtype=torch.float32)
    out_ref = ref(x_nchw)
    out_fast_nchw = fast(x_nchw)
    out_fast_nhwc = fast(x_nchw.to(memory_format=torch.channels_last))

    assert (out_ref - out_fast_nchw).abs().max().item() < 1e-5
    assert (out_ref - out_fast_nhwc).abs().max().item() < 1e-5
    assert out_fast_nhwc.is_contiguous(memory_format=torch.channels_last)


def test_infer_tiled_batched_identity_and_shift_on_non_divisible_shape() -> None:
    torch.manual_seed(7)
    image = torch.rand((3, 3, 130, 190), dtype=torch.float32)

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
        tile_batch_size=4,
    )
    assert out_id.shape == image.shape
    assert (out_id - image).abs().max().item() < 1e-5

    out_shift = infer_tiled(
        shift,
        image,
        tile_size=64,
        tile_overlap=16,
        multiple_of=4,
        tile_batch_size=4,
    )
    assert out_shift.shape == image.shape
    assert (out_shift - (image + 0.125)).abs().max().item() < 1e-5


def test_infer_tiled_full_frame_padding() -> None:
    image = torch.rand((2, 3, 35, 51), dtype=torch.float32)

    def double_identity(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == 2
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


def test_resolve_auto_tile_size() -> None:
    cuda_12gb = DeviceInfo(
        device=torch.device("cpu"),
        backend="cuda",
        name="RTX 4070 SUPER",
        default_fp16=True,
        total_memory_bytes=12 * 1024**3,
    )
    cuda_4gb = DeviceInfo(
        device=torch.device("cpu"),
        backend="cuda",
        name="Small GPU",
        default_fp16=True,
        total_memory_bytes=4 * 1024**3,
    )
    cpu_dev = DeviceInfo(device=torch.device("cpu"), backend="cpu", name="CPU", default_fp16=False)

    assert resolve_auto_tile_size(cuda_12gb, 1080, 1920, batch_size=1, dtype=torch.float16) == 0
    assert resolve_auto_tile_size(cuda_12gb, 1080, 1920, batch_size=2, dtype=torch.float16) == 0
    assert resolve_auto_tile_size(cuda_4gb, 1080, 1920, batch_size=1, dtype=torch.float16) == 512
    assert resolve_auto_tile_size(cpu_dev, 1080, 1920, batch_size=1, dtype=torch.float32) == 512
