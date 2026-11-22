"""Configuration, presets, and geometry resolution for ruby.ai."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

PresetName = Literal["natural", "crisp", "low-light", "passthrough"]

ALLOWED_CODECS = frozenset({"libx264", "h264_nvenc", "h264_videotoolbox"})
ALLOWED_X264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
)

PRESETS: dict[PresetName, dict[str, Any]] = {
    "natural": {
        "model": "de-h264",
        "temporal_denoise": 0.02,
        "ai_strength": 0.45,
        "sharpen": 0.15,
        "grain": 2,
    },
    "crisp": {
        "model": "de-h264",
        "temporal_denoise": 0.03,
        "ai_strength": 0.65,
        "sharpen": 0.25,
        "grain": 2,
    },
    "low-light": {
        "model": "denoise",
        "temporal_denoise": 0.05,
        "ai_strength": 0.75,
        "sharpen": 0.20,
        "grain": 3,
    },
    "passthrough": {
        "model": "de-h264",
        "temporal_denoise": 0.02,
        "ai_strength": 0.0,
        "sharpen": 0.15,
        "grain": 0,
    },
}


@dataclass(slots=True)
class EnhanceConfig:
    """Validated configuration for the 5-stage restoration pipeline."""

    model: str = "de-h264"
    device: str = "auto"
    fp16: bool | None = None
    max_short_edge: int = 1080
    temporal_denoise: float = 0.03
    ai_strength: float = 0.65
    sharpen: float = 0.25
    grain: int = 2
    tile_size: int = 512
    tile_overlap: int = 32
    crf: int = 16
    codec: str = "libx264"
    x264_preset: str = "slow"
    allow_unsafe_pickle: bool = False

    def __post_init__(self) -> None:
        if not self.model or not self.model.strip():
            raise ValueError("model must be a non-empty registry key or file path")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported device '{self.device}' (expected auto, cuda, mps, cpu)")
        if not (2 <= self.max_short_edge <= 8192):
            raise ValueError("max_short_edge must be between 2 and 8192")
        if not (0.0 <= self.temporal_denoise <= 0.25):
            raise ValueError("temporal_denoise must be between 0.0 and 0.25")
        if not (0.0 <= self.ai_strength <= 1.0):
            raise ValueError("ai_strength must be between 0.0 and 1.0")
        if not (0.0 <= self.sharpen <= 1.0):
            raise ValueError("sharpen must be between 0.0 and 1.0")
        if not (0 <= self.grain <= 10):
            raise ValueError("grain must be between 0 and 10")
        if self.tile_size < 0:
            raise ValueError("tile_size must be >= 0 (0 disables tiling)")
        if self.tile_overlap < 0:
            raise ValueError("tile_overlap must be >= 0")
        if self.tile_size > 0 and self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be strictly smaller than tile_size")
        if not (0 <= self.crf <= 51):
            raise ValueError("crf must be between 0 and 51")
        if self.codec not in ALLOWED_CODECS:
            raise ValueError(
                f"Unsupported codec '{self.codec}' (allowed: {sorted(ALLOWED_CODECS)})"
            )
        if self.x264_preset not in ALLOWED_X264_PRESETS:
            raise ValueError(
                f"Unsupported x264_preset '{self.x264_preset}' "
                f"(allowed: {sorted(ALLOWED_X264_PRESETS)})"
            )

    @classmethod
    def from_preset(cls, preset: PresetName, **overrides: Any) -> EnhanceConfig:
        """Create an EnhanceConfig from a named preset with optional non-None overrides."""
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}' (allowed: {sorted(PRESETS)})")
        base = cls(**PRESETS[preset])
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        return replace(base, **clean_overrides)


def _round_even(value: float) -> int:
    """Round a positive dimension to the nearest even integer >= 2."""
    rounded = int(round(value / 2.0) * 2)
    return max(2, rounded)


def resolve_target_dimensions(
    src_width: int,
    src_height: int,
    max_short_edge: int,
) -> tuple[int, int]:
    """Compute even target dimensions capping the short edge without upscaling."""
    if src_width < 2 or src_height < 2:
        raise ValueError(f"Invalid source dimensions: {src_width}x{src_height}")
    if max_short_edge < 2:
        raise ValueError(f"Invalid max_short_edge: {max_short_edge}")

    short_edge = min(src_width, src_height)
    if short_edge <= max_short_edge:
        return _round_even(src_width), _round_even(src_height)

    scale = max_short_edge / short_edge
    return _round_even(src_width * scale), _round_even(src_height * scale)
