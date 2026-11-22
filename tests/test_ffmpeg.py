"""Tests for FFmpeg filter builders, rotation handling, and video probing."""

import subprocess
from pathlib import Path

from ruby_ai.config import EnhanceConfig
from ruby_ai.ffmpeg import (
    build_decode_filter,
    build_encode_filter,
    parse_display_dimensions,
    probe_video,
    require_binary,
)


def test_build_decode_filter_downscales_and_denoises() -> None:
    cfg = EnhanceConfig(temporal_denoise=0.03)
    vf = build_decode_filter(cfg, target_w=1080, target_h=1920, src_w=2160, src_h=3840)
    assert "atadenoise=0a=0.0300:0b=0.0600" in vf
    assert "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int" in vf
    assert vf.endswith("format=rgb24")


def test_build_decode_filter_skips_redundant_scale_and_denoise() -> None:
    cfg = EnhanceConfig(temporal_denoise=0.0)
    vf = build_decode_filter(cfg, target_w=1080, target_h=1920, src_w=1080, src_h=1920)
    assert vf == "format=rgb24"


def test_build_encode_filter_combinations() -> None:
    cfg = EnhanceConfig(sharpen=0.25, grain=2)
    vf = build_encode_filter(cfg)
    assert "cas=strength=0.250" in vf
    assert "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" in vf
    assert vf.index("format=yuv420p") < vf.index("noise=c0s=2:c0f=t+u")

    plain_cfg = EnhanceConfig(sharpen=0.0, grain=0)
    plain_vf = build_encode_filter(plain_cfg)
    assert "cas=" not in plain_vf
    assert "noise=" not in plain_vf
    assert plain_vf.endswith("format=yuv420p")


def test_parse_display_dimensions_handles_rotation() -> None:
    assert parse_display_dimensions(3840, 2160, {"tags": {"rotate": "90"}}) == (2160, 3840, 90)
    assert parse_display_dimensions(
        3840,
        2160,
        {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
    ) == (2160, 3840, -90)
    assert parse_display_dimensions(3840, 2160, {"tags": {"rotate": "180"}}) == (3840, 2160, 180)
    assert parse_display_dimensions(2160, 3840, {}) == (2160, 3840, 0)


def test_probe_video_synthetic_clip(tmp_path: Path) -> None:
    ffmpeg_bin = require_binary("ffmpeg")
    clip = tmp_path / "sample.mp4"
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x480:rate=24:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        check=True,
    )

    info = probe_video(clip)
    assert info.display_width == 320
    assert info.display_height == 480
    assert info.fps_num == 24
    assert info.fps_den == 1
    assert info.has_audio is True
    assert info.codec_name == "h264"
    assert info.frame_count >= 10
