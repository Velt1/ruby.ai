"""Fast 3-point image-content video analysis (Topaz Estimate equivalent)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ruby_ai.config import PRESETS, PresetName
from ruby_ai.ffmpeg import (
    probe_bitrate_kbps,
    probe_video,
    require_binary,
    subprocess_creation_flags,
)


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
    noise_score: float = 0.0
    sharpness_score: float = 0.5
    compression_score: float = 0.0


def _extract_sample_rgb_frame(
    ffmpeg_bin: str,
    path: Path,
    timestamp_sec: float,
    sample_w: int = 384,
) -> np.ndarray | None:
    """Fast keyframe-seek single RGB24 frame decode for content analysis."""
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        str(path),
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-vf",
        f"scale={sample_w}:-2:flags=bilinear",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            timeout=15,
            creationflags=subprocess_creation_flags(),
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None

    total_pixels = len(proc.stdout) // 3
    if total_pixels < sample_w * 16:
        return None
    sample_h = total_pixels // sample_w
    expected_bytes = sample_w * sample_h * 3
    if expected_bytes <= 0 or len(proc.stdout) < expected_bytes:
        return None

    return (
        np.frombuffer(proc.stdout[:expected_bytes], dtype=np.uint8)
        .reshape((sample_h, sample_w, 3))
        .astype(np.float32)
    )


def _compute_frame_metrics(rgb: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute (luma_255, sat_255, noise_01, sharpness_01, blockiness_01) from RGB float."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 1e-3, ((max_c - min_c) / np.maximum(max_c, 1.0)) * 255.0, 0.0)

    avg_luma = float(np.mean(y))
    avg_sat = float(np.mean(sat))

    # Spatial gradients (edge / sharpness energy)
    gx = np.abs(y[:, 1:] - y[:, :-1])
    gy = np.abs(y[1:, :] - y[:-1, :])
    edge_mag = 0.5 * (gx[:-1, :] + gy[:, :-1])
    sharpness_score = float(np.clip(np.percentile(edge_mag, 90) / 38.0, 0.0, 1.0))

    # High-pass Laplacian MAD in flat low-gradient areas (robust noise floor estimator)
    lap = (
        4.0 * y[1:-1, 1:-1]
        - y[:-2, 1:-1]
        - y[2:, 1:-1]
        - y[1:-1, :-2]
        - y[1:-1, 2:]
    )
    flat_thresh = np.percentile(edge_mag, 45) + 0.5
    flat_mask = edge_mag[:-1, :-1] <= flat_thresh
    flat_lap = np.abs(lap[flat_mask]) if np.any(flat_mask) else np.abs(lap)
    mad_noise = float(np.median(flat_lap))
    noise_score = float(np.clip((mad_noise - 1.2) / 14.0, 0.0, 1.0))

    # 8x8 DCT block-boundary discontinuity ratio (captures macroblock compression artifacts)
    h, w = y.shape
    block_ratios: list[float] = []
    if w >= 32:
        col_diffs = np.mean(np.abs(y[:, 1:] - y[:, :-1]), axis=0)
        boundary_cols = col_diffs[7::8]
        interior_mask = np.ones(len(col_diffs), dtype=bool)
        interior_mask[7::8] = False
        interior_cols = col_diffs[interior_mask]
        if len(boundary_cols) > 0 and len(interior_cols) > 0:
            block_ratios.append(
                float(np.mean(boundary_cols) / max(0.5, float(np.mean(interior_cols))))
            )
    if h >= 32:
        row_diffs = np.mean(np.abs(y[1:, :] - y[:-1, :]), axis=1)
        boundary_rows = row_diffs[7::8]
        interior_mask_r = np.ones(len(row_diffs), dtype=bool)
        interior_mask_r[7::8] = False
        interior_rows = row_diffs[interior_mask_r]
        if len(boundary_rows) > 0 and len(interior_rows) > 0:
            block_ratios.append(
                float(np.mean(boundary_rows) / max(0.5, float(np.mean(interior_rows))))
            )

    block_ratio = float(np.mean(block_ratios)) if block_ratios else 1.0
    blockiness_01 = float(np.clip((block_ratio - 1.05) / 0.45, 0.0, 1.0))
    return avg_luma, avg_sat, noise_score, sharpness_score, blockiness_01


def _sample_content_metrics(
    path: Path,
    duration: float,
) -> tuple[float, float, float, float, float]:
    """Sample 3 representative keyframes quickly and average content metrics."""
    ffmpeg_bin = require_binary("ffmpeg")
    if duration > 2.5:
        timestamps = [duration * 0.15, duration * 0.50, duration * 0.85]
    else:
        timestamps = [max(0.0, duration * 0.35)]

    samples: list[tuple[float, float, float, float, float]] = []
    for ts in timestamps:
        frame = _extract_sample_rgb_frame(ffmpeg_bin, path, ts)
        if frame is not None:
            samples.append(_compute_frame_metrics(frame))

    if not samples:
        return 110.0, 40.0, 0.15, 0.5, 0.1

    arr = np.mean(np.array(samples, dtype=np.float32), axis=0)
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4])


def analyze_video(path: Path) -> VideoAnalysis:
    """Analyze clip brightness, noise, sharpness, compression artifacts, HDR, and FPS."""
    info = probe_video(path)
    fps = info.fps_num / info.fps_den if info.fps_den else 30.0
    try:
        total_kbps, video_kbps = probe_bitrate_kbps(path)
    except Exception:
        total_kbps, video_kbps = 0.0, 0.0

    effective_kbps = video_kbps if video_kbps > 0.0 else total_kbps
    pixels_per_sec = max(1.0, float(info.display_width * info.display_height) * fps)
    bits_per_pixel = (effective_kbps * 1000.0) / pixels_per_sec if effective_kbps > 0.0 else 0.20

    avg_luma, avg_sat, noise_score, sharpness_score, blockiness_01 = _sample_content_metrics(
        path, info.duration
    )
    bpp_comp = (
        float(np.clip((0.12 - bits_per_pixel) / 0.09, 0.0, 1.0))
        if effective_kbps > 0.0
        else 0.2
    )
    compression_score = float(np.clip(0.55 * bpp_comp + 0.45 * blockiness_01, 0.0, 1.0))

    is_low_light = avg_luma < 68.0 or (avg_luma < 85.0 and noise_score >= 0.45)
    is_heavily_compressed = (
        (effective_kbps > 0.0 and bits_per_pixel < 0.08) or compression_score >= 0.50
    )
    is_small_source = min(info.display_width, info.display_height) <= 720
    is_hdr = info.is_hdr
    is_high_fps = fps > 35.0

    reasons: list[str] = []
    if is_low_light:
        recommended_preset: PresetName = "low-light"
        overrides: dict[str, Any] = {
            **PRESETS["low-light"],
            "ai_strength": round(0.68 + 0.20 * noise_score, 2),
            "temporal_denoise": round(0.03 + 0.04 * noise_score, 3),
            "recover_details": round(max(0.08, 0.22 - 0.14 * noise_score), 2),
            "temporal_consistency": 0.35,
            "sharpen": 0.30,
            "clarity": 0.16,
            "deband": True,
            "grain": 3,
        }
        reasons.append(
            f"Nacht-/Rauschszene erkannt (Ø Helligkeit {avg_luma:.0f}/255, "
            f"Rauschen {round(noise_score * 100)}%)."
        )
    elif is_heavily_compressed:
        recommended_preset = "social-crisp"
        overrides = {
            **PRESETS["social-crisp"],
            "ai_strength": round(0.72 + 0.16 * compression_score, 2),
            "temporal_denoise": round(0.03 + 0.02 * noise_score, 3),
            "recover_details": 0.16,
            "dehalo": 0.20,
            "temporal_consistency": 0.28,
            "sharpen": 0.42,
            "clarity": 0.22,
            "deband": True,
            "grain": 2,
        }
        reasons.append(
            f"Kompressionsartefakte erkannt ({bits_per_pixel:.3f} bpp, "
            f"Block/Artefakt-Score {round(compression_score * 100)}%) – "
            "verstärkte De-Blocking-Restauration."
        )
    elif is_small_source and sharpness_score < 0.55:
        recommended_preset = "upscale-2x"
        overrides = dict(PRESETS["upscale-2x"])
        reasons.append(
            f"Niedrige Quellauflösung ({info.display_width}×{info.display_height}) – "
            "2× AI Upscaler für scharfes 1080p/4K empfohlen."
        )
    else:
        recommended_preset = "social-crisp"
        softness = 1.0 - sharpness_score
        overrides = {
            **PRESETS["social-crisp"],
            "sharpen": round(float(np.clip(0.32 + 0.18 * softness, 0.25, 0.50)), 2),
            "recover_details": round(float(np.clip(0.18 + 0.12 * sharpness_score, 0.15, 0.30)), 2),
        }
        reasons.append(
            f"Saubere Quelle ({info.display_width}×{info.display_height}, "
            f"Schärfe {round(sharpness_score * 100)}%, Rauschen {round(noise_score * 100)}%) – "
            "Social Ultra-Crisp mit WaveletFix & Detail-Recovery."
        )

    if is_hdr:
        reasons.append(
            f"HDR ({info.color_transfer}, {info.bit_depth}-Bit) erkannt – "
            "16-Bit Pipe + Rec.709 Tone-Mapping aktiv."
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
        noise_score=round(noise_score, 3),
        sharpness_score=round(sharpness_score, 3),
        compression_score=round(compression_score, 3),
    )
