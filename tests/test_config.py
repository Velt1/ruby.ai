"""Tests for configuration presets, validation, and geometry resolution."""

import pytest

from ruby_ai.config import PRESETS, EnhanceConfig, resolve_target_dimensions


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


def test_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        EnhanceConfig(temporal_denoise=0.5)
    with pytest.raises(ValueError):
        EnhanceConfig(ai_strength=1.2)
    with pytest.raises(ValueError):
        EnhanceConfig(sharpen=-0.1)
    with pytest.raises(ValueError):
        EnhanceConfig(grain=12)
    with pytest.raises(ValueError):
        EnhanceConfig(tile_size=64, tile_overlap=64)
    with pytest.raises(ValueError):
        EnhanceConfig(codec="prores")
    with pytest.raises(ValueError):
        EnhanceConfig(x264_preset="invalid")


def test_resolve_target_dimensions() -> None:
    assert resolve_target_dimensions(2160, 3840, 1080) == (1080, 1920)
    assert resolve_target_dimensions(3840, 2160, 1080) == (1920, 1080)
    assert resolve_target_dimensions(1080, 1920, 1080) == (1080, 1920)
    assert resolve_target_dimensions(720, 1280, 1080) == (720, 1280)
    assert resolve_target_dimensions(1081, 1921, 1080) == (1080, 1920)
    w, h = resolve_target_dimensions(320, 480, 160)
    assert (w, h) == (160, 240)
    assert w % 2 == 0 and h % 2 == 0
