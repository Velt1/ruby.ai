"""Configuration, presets, social export targets, and geometry resolution for ruby.ai."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

PresetName = Literal[
    "social-crisp",
    "natural",
    "crisp",
    "low-light",
    "fast",
    "passthrough",
]

ALLOWED_CODECS = frozenset({"auto", "libx264", "h264_nvenc", "h264_videotoolbox"})
ALLOWED_NVENC_PRESETS = frozenset({"p1", "p2", "p3", "p4", "p5", "p6", "p7"})
ALLOWED_HWACCELS = frozenset({"none", "auto", "cuda"})
ALLOWED_EXPORT_TARGETS = frozenset({"social-1080p", "master-4k", "custom"})
ALLOWED_AUDIO_SAMPLE_RATES = frozenset({44100, 48000})
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

EXPORT_TARGETS: dict[str, dict[str, Any]] = {
    "social-1080p": {
        "label": "TikTok & Instagram (1080p)",
        "badge": "Empfohlen",
        "max_short_edge": 1080,
        "maxrate_kbps_30fps": 14000,
        "maxrate_kbps_60fps": 20000,
        "gop_seconds": 2.0,
        "audio_sample_rate": 48000,
        "description": (
            "Exakt auf die TikTok/Reels Ingest-Pipeline abgestimmt "
            "(1080p, Closed GOP, Rec.709, 48 kHz AAC)."
        ),
    },
    "master-4k": {
        "label": "Ultra 4K Master",
        "badge": "Desktop-Upload",
        "max_short_edge": 2160,
        "maxrate_kbps_30fps": 24000,
        "maxrate_kbps_60fps": 32000,
        "gop_seconds": 2.0,
        "audio_sample_rate": 48000,
        "description": (
            "4K-Master mit sauberem Bitrate-Limit für Desktop-Web-Uploads oder Schnittprogramme."
        ),
    },
    "custom": {
        "label": "Benutzerdefiniert",
        "badge": "Profi",
        "max_short_edge": 1080,
        "maxrate_kbps_30fps": None,
        "maxrate_kbps_60fps": None,
        "gop_seconds": 2.0,
        "audio_sample_rate": 48000,
        "description": "Freie Kontrolle über Auflösung und Bitrate ohne VBV-Deckel.",
    },
}

PRESETS: dict[PresetName, dict[str, Any]] = {
    "social-crisp": {
        "model": "de-h264",
        "temporal_denoise": 0.03,
        "ai_strength": 0.70,
        "sharpen": 0.42,
        "clarity": 0.22,
        "deband": True,
        "grain": 2,
    },
    "natural": {
        "model": "de-h264",
        "temporal_denoise": 0.02,
        "ai_strength": 0.45,
        "sharpen": 0.15,
        "clarity": 0.0,
        "deband": False,
        "grain": 2,
    },
    "crisp": {
        "model": "de-h264",
        "temporal_denoise": 0.03,
        "ai_strength": 0.65,
        "sharpen": 0.25,
        "clarity": 0.0,
        "deband": False,
        "grain": 2,
    },
    "low-light": {
        "model": "denoise",
        "temporal_denoise": 0.05,
        "ai_strength": 0.75,
        "sharpen": 0.30,
        "clarity": 0.15,
        "deband": True,
        "grain": 3,
    },
    "fast": {
        "model": "superscale-span",
        "temporal_denoise": 0.02,
        "ai_strength": 0.65,
        "sharpen": 0.35,
        "clarity": 0.18,
        "deband": False,
        "grain": 2,
    },
    "passthrough": {
        "model": "de-h264",
        "temporal_denoise": 0.02,
        "ai_strength": 0.0,
        "sharpen": 0.15,
        "clarity": 0.0,
        "deband": False,
        "grain": 0,
    },
}

PROFILE_METADATA: list[dict[str, Any]] = [
    {
        "key": "social-crisp",
        "label": "Social Ultra-Crisp",
        "badge": "Empfohlen",
        "style": "Proteus-Stil",
        "tagline": "Maximaler 4K-Look für TikTok & Reels nach Kompression.",
        "primary": True,
    },
    {
        "key": "fast",
        "label": "Turbo Social",
        "badge": "~7 fps",
        "style": "Artemis-Stil",
        "tagline": "Sehr schneller Export mit knackiger Kanten- & Detail-Schärfe.",
        "primary": True,
    },
    {
        "key": "low-light",
        "label": "Nacht & Low-Light",
        "badge": "Nyx-Stil",
        "style": "Nyx-Stil",
        "tagline": "Ruhige Schatten ohne Blockbildung für dunkle & verrauschte Clips.",
        "primary": True,
    },
    {
        "key": "natural",
        "label": "Natürlich & Gesichter",
        "badge": "Gaia-Stil",
        "style": "Gaia-Stil",
        "tagline": "Zurückhaltende Optimierung für natürliche Hauttöne.",
        "primary": True,
    },
    {
        "key": "crisp",
        "label": "Classic Crisp",
        "badge": "CLI",
        "style": "Standard",
        "tagline": "Klassisches CLI-Preset ohne Clarity/Deband.",
        "primary": False,
    },
    {
        "key": "passthrough",
        "label": "Nur FFmpeg (ohne KI)",
        "badge": "Schnell",
        "style": "Passthrough",
        "tagline": "Reines Lanczos-Supersampling ohne neuronales Modell.",
        "primary": False,
    },
]


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
    clarity: float = 0.0
    deband: bool = False
    grain: int = 2
    hdr_to_sdr: bool = True
    export_target: str = "social-1080p"
    maxrate_kbps: int | None = None
    gop_seconds: float = 2.0
    cfr: bool = True
    audio_sample_rate: int = 48000
    tile_size: int | None = None
    tile_overlap: int = 32
    tile_batch_size: int = 4
    batch_size: int = 1
    crf: int = 16
    codec: str = "libx264"
    x264_preset: str = "slow"
    nvenc_preset: str = "p6"
    hwaccel: str = "none"
    optimize_gpu: bool = True
    compile_model: bool = False
    async_io: bool = True
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
        if not (0.0 <= self.clarity <= 1.0):
            raise ValueError("clarity must be between 0.0 and 1.0")
        if not (0 <= self.grain <= 10):
            raise ValueError("grain must be between 0 and 10")
        if self.export_target not in ALLOWED_EXPORT_TARGETS:
            raise ValueError(
                f"Unsupported export_target '{self.export_target}' "
                f"(allowed: {sorted(ALLOWED_EXPORT_TARGETS)})"
            )
        if self.maxrate_kbps is not None and not (500 <= self.maxrate_kbps <= 200_000):
            raise ValueError("maxrate_kbps must be between 500 and 200000")
        if not (0.0 <= self.gop_seconds <= 10.0):
            raise ValueError("gop_seconds must be between 0.0 and 10.0")
        if self.audio_sample_rate not in ALLOWED_AUDIO_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported audio_sample_rate '{self.audio_sample_rate}' "
                f"(allowed: {sorted(ALLOWED_AUDIO_SAMPLE_RATES)})"
            )
        if self.tile_overlap < 0:
            raise ValueError("tile_overlap must be >= 0")
        if self.tile_size is not None:
            if self.tile_size < 0:
                raise ValueError(
                    "tile_size must be >= 0 (0 disables tiling, None selects automatically)"
                )
            if self.tile_size > 0 and self.tile_overlap >= self.tile_size:
                raise ValueError("tile_overlap must be strictly smaller than tile_size")
        if not (1 <= self.tile_batch_size <= 64):
            raise ValueError("tile_batch_size must be between 1 and 64")
        if not (1 <= self.batch_size <= 16):
            raise ValueError("batch_size must be between 1 and 16")
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
        if self.nvenc_preset not in ALLOWED_NVENC_PRESETS:
            raise ValueError(
                f"Unsupported nvenc_preset '{self.nvenc_preset}' "
                f"(allowed: {sorted(ALLOWED_NVENC_PRESETS)})"
            )
        if self.hwaccel not in ALLOWED_HWACCELS:
            raise ValueError(
                f"Unsupported hwaccel '{self.hwaccel}' (allowed: {sorted(ALLOWED_HWACCELS)})"
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
