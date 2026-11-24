"""Fast automatic video analysis (Topaz Estimate equivalent) for preset recommendation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruby_ai.config import PRESETS, PresetName
from ruby_ai.ffmpeg import probe_bitrate_kbps, probe_video, require_binary


@dataclass(frozen=True, slots=True)
class VideoAnalysis:
    """Detected clip characteristics and recommended social-4K settings."""

    path: Path
    avg_luma: float
    avg_saturation: float
    bits_per_pixel: float
    is_low_light: bool
    is_heavily_compressed: bool
    is_hdr: bool
    is_high_fps: bool
    recommended_preset: PresetName
    recommended_export_target: str
    recommended_overrides: dict[str, Any]
    reasons: list[str]


def _sample_signalstats(
    path: Path,
    duration: float,
) -> tuple[float, float]:
    """Sample representative frames quickly to compute average YAVG and SATAVG."""
    ffmpeg_bin = require_binary("ffmpeg")
    window = min(12.0, duration) if duration > 0.0 else 2.0
    start = min(1.0, duration * 0.15) if duration > 3.0 else 0.0
    sample_fps = min(4.0, max(0.5, 4.0 / max(0.5, window)))

    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{window:.3f}",
        "-i",
        str(path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"fps={sample_fps:.3f},scale=320:-2,signalstats,metadata=mode=print:file=-",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return 110.0, 40.0

    yavgs: list[float] = []
    satavgs: list[float] = []
    for line in proc.stdout.splitlines():
        if "lavfi.signalstats.YAVG=" in line:
            try:
                yavgs.append(float(line.split("=", 1)[1]))
            except ValueError:
                pass
        elif "lavfi.signalstats.SATAVG=" in line:
            try:
                satavgs.append(float(line.split("=", 1)[1]))
            except ValueError:
                pass

    avg_luma = sum(yavgs) / len(yavgs) if yavgs else 110.0
    avg_sat = sum(satavgs) / len(satavgs) if satavgs else 40.0
    return avg_luma, avg_sat


def analyze_video(path: Path) -> VideoAnalysis:
    """Analyze clip brightness, compression density, HDR status, and FPS."""
    info = probe_video(path)
    fps = info.fps_num / info.fps_den if info.fps_den else 30.0
    try:
        total_kbps, video_kbps = probe_bitrate_kbps(path)
    except Exception:
        total_kbps, video_kbps = 0.0, 0.0

    effective_kbps = video_kbps if video_kbps > 0.0 else total_kbps
    pixels_per_sec = max(1.0, float(info.display_width * info.display_height) * fps)
    bits_per_pixel = (effective_kbps * 1000.0) / pixels_per_sec if effective_kbps > 0.0 else 0.20

    avg_luma, avg_sat = _sample_signalstats(path, info.duration)
    is_low_light = avg_luma < 68.0
    is_heavily_compressed = effective_kbps > 0.0 and bits_per_pixel < 0.08
    is_hdr = info.is_hdr
    is_high_fps = fps > 35.0

    reasons: list[str] = []
    if is_low_light:
        recommended_preset: PresetName = "low-light"
        overrides: dict[str, Any] = {
            **PRESETS["low-light"],
            "sharpen": 0.32,
            "clarity": 0.18,
            "deband": True,
            "grain": 3,
        }
        reasons.append(f"Nacht-/Low-Light-Szene erkannt (Ø Helligkeit {avg_luma:.0f}/255).")
    elif is_heavily_compressed:
        recommended_preset = "social-crisp"
        overrides = {
            **PRESETS["social-crisp"],
            "ai_strength": 0.78,
            "temporal_denoise": 0.04,
            "sharpen": 0.44,
            "clarity": 0.24,
            "deband": True,
            "grain": 2,
        }
        reasons.append(
            f"Stärkere Quell-Kompression erkannt ({bits_per_pixel:.3f} bpp) – "
            "verstärkte Artefakt-Korrektur."
        )
    else:
        recommended_preset = "social-crisp"
        overrides = dict(PRESETS["social-crisp"])
        reasons.append(
            f"Saubere Quelle ({info.display_width}×{info.display_height}, "
            f"Ø Helligkeit {avg_luma:.0f}/255) – Social Ultra-Crisp für maximalen 4K-Look."
        )

    if is_hdr:
        reasons.append(
            f"HDR ({info.color_transfer}) erkannt – automatisches Rec.709 SDR Tone-Mapping aktiv."
        )
    if is_high_fps:
        reasons.append(
            f"{fps:.0f} fps erkannt – Social-Ingest-Limit automatisch auf 20 Mbps erhöht."
        )

    return VideoAnalysis(
        path=path,
        avg_luma=round(avg_luma, 1),
        avg_saturation=round(avg_sat, 1),
        bits_per_pixel=round(bits_per_pixel, 4),
        is_low_light=is_low_light,
        is_heavily_compressed=is_heavily_compressed,
        is_hdr=is_hdr,
        is_high_fps=is_high_fps,
        recommended_preset=recommended_preset,
        recommended_export_target="social-1080p",
        recommended_overrides=overrides,
        reasons=reasons,
    )
