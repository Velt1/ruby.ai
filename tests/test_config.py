"""Tests for configuration presets, social export targets, validation, and geometry."""

import pytest

from ruby_ai.config import (
    EXPORT_TARGETS,
    PRESETS,
    PROFILE_METADATA,
    EnhanceConfig,
    resolve_target_dimensions,
)


def test_presets_resolve_and_ignore_none_overrides() -> None:
    for preset_name in PRESETS:
        cfg = EnhanceConfig.from_preset(
            preset_name,  # type: ignore[arg-type]
            model=None,
            ai_strength=None,
            crf=14,
        )
        assert cfg.model == PRESETS[preset_name]["model"]
        assert cfg.ai_strength == PRESETS[preset_name]["ai_strength"]
        assert cfg.crf == 14

    social_cfg = EnhanceConfig.from_preset("social-crisp")
    assert social_cfg.model == "de-h264"
    assert social_cfg.clarity > 0.0
    assert social_cfg.deband is True
    assert social_cfg.export_target == "social-1080p"

    fast_cfg = EnhanceConfig.from_preset("fast")
    assert fast_cfg.model == "superscale-span"
    assert fast_cfg.tile_size is None

    profile_keys = {p["key"] for p in PROFILE_METADATA}
    assert set(PRESETS.keys()).issubset(profile_keys)
    assert {"social-1080p", "master-4k", "custom"}.issubset(EXPORT_TARGETS.keys())


def test_config_allows_auto_tile_size_and_auto_codec() -> None:
    cfg = EnhanceConfig(tile_size=None, codec="auto", batch_size=2, tile_batch_size=4)
    assert cfg.tile_size is None
    assert cfg.codec == "auto"

    full_frame = EnhanceConfig.from_preset("crisp", tile_size=0)
    assert full_frame.tile_size == 0


def test_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        EnhanceConfig(temporal_denoise=0.5)
    with pytest.raises(ValueError):
        EnhanceConfig(ai_strength=1.2)
    with pytest.raises(ValueError):
        EnhanceConfig(sharpen=-0.1)
    with pytest.raises(ValueError):
        EnhanceConfig(clarity=1.5)
    with pytest.raises(ValueError):
        EnhanceConfig(grain=12)
    with pytest.raises(ValueError):
        EnhanceConfig(export_target="youtube-8k")
    with pytest.raises(ValueError):
        EnhanceConfig(maxrate_kbps=100)
    with pytest.raises(ValueError):
        EnhanceConfig(gop_seconds=-1.0)
    with pytest.raises(ValueError):
        EnhanceConfig(audio_sample_rate=22050)
    with pytest.raises(ValueError):
        EnhanceConfig(tile_size=-1)
    with pytest.raises(ValueError):
        EnhanceConfig(tile_size=64, tile_overlap=64)
    with pytest.raises(ValueError):
        EnhanceConfig(batch_size=0)
    with pytest.raises(ValueError):
        EnhanceConfig(batch_size=32)
    with pytest.raises(ValueError):
        EnhanceConfig(tile_batch_size=0)
    with pytest.raises(ValueError):
        EnhanceConfig(codec="prores")
    with pytest.raises(ValueError):
        EnhanceConfig(x264_preset="invalid")
    with pytest.raises(ValueError):
        EnhanceConfig(nvenc_preset="p9")
    with pytest.raises(ValueError):
        EnhanceConfig(hwaccel="vulkan")


def test_resolve_target_dimensions() -> None:
    assert resolve_target_dimensions(2160, 3840, 1080) == (1080, 1920)
    assert resolve_target_dimensions(3840, 2160, 1080) == (1920, 1080)
    assert resolve_target_dimensions(1080, 1920, 1080) == (1080, 1920)
    assert resolve_target_dimensions(720, 1280, 1080) == (720, 1280)
    assert resolve_target_dimensions(1081, 1921, 1080) == (1080, 1920)
    w, h = resolve_target_dimensions(320, 480, 160)
    assert (w, h) == (160, 240)
    assert w % 2 == 0 and h % 2 == 0
