"""Open-source video restoration, upscaling, and supersampling engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ruby_ai.config import EnhanceConfig, PresetName

if TYPE_CHECKING:
    from ruby_ai.pipeline import enhance_video

__version__ = "0.3.0"

__all__ = [
    "EnhanceConfig",
    "PresetName",
    "__version__",
    "enhance_video",
]


def __getattr__(name: str) -> Any:
    if name == "enhance_video":
        from ruby_ai.pipeline import enhance_video

        return enhance_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
