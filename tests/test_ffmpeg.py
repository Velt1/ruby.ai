"""Tests for FFmpeg filter builders, HDR->SDR, Social-Ingest flags, and probing."""

import subprocess
from pathlib import Path

from ruby_ai.config import ALLOWED_CODECS, EnhanceConfig
from ruby_ai.ffmpeg import (
    VideoInfo,
    _codec_args,
    build_decode_command,
    build_decode_filter,
    build_encode_command,
    build_encode_filter,
    parse_display_dimensions,
    probe_video,
    require_binary,
    resolve_codec,
)


def test_build_decode_filter_downscales_denoises_and_hdr_tonemap() -> None:
    cfg = EnhanceConfig(temporal_denoise=0.03, hdr_to_sdr=True)
    vf = build_decode_filter(
        cfg,
        target_w=1080,
        target_h=1920,
        src_w=2160,
        src_h=3840,
        is_hdr=True,
        color_transfer="arib-std-b67",
    )
    assert "zscale=pin=bt2020" in vf
    assert "tonemap=tonemap=hable" in vf
    assert "atadenoise=0a=0.0300:0b=0.0600" in vf
    assert "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int" in vf
    assert vf.endswith("format=rgb24")


def test_build_decode_filter_skips_redundant_scale_and_denoise() -> None:
    cfg = EnhanceConfig(temporal_denoise=0.0)
    vf = build_decode_filter(cfg, target_w=1080, target_h=1920, src_w=1080, src_h=1920)
    assert vf == "format=rgb24"


def test_build_encode_filter_combinations() -> None:
    cfg = EnhanceConfig.from_preset("social-crisp")
    vf = build_encode_filter(cfg)
    assert "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" in vf
    assert "deband=1thr=0.02" in vf
    assert "cas=strength=0.420:planes=1" in vf
    assert "unsharp=lx=13:ly=13" in vf
    assert "eq=contrast=" in vf
    assert "noise=c0s=2:c0f=t+u" in vf
    assert vf.index("format=yuv420p") < vf.index("cas=") < vf.index("noise=c0s=2:c0f=t+u")

    plain_cfg = EnhanceConfig(sharpen=0.0, clarity=0.0, deband=False, grain=0)
    plain_vf = build_encode_filter(plain_cfg)
    assert "cas=" not in plain_vf
    assert "deband=" not in plain_vf
    assert "unsharp=" not in plain_vf
    assert "noise=" not in plain_vf
    assert plain_vf.endswith("format=yuv420p")


def test_nvenc_and_social_ingest_encode_flags() -> None:
    cfg = EnhanceConfig(codec="h264_nvenc", nvenc_preset="p7", crf=15, export_target="social-1080p")
    args = _codec_args(cfg, target_w=1080, target_h=1920, fps=30.0)
    joined = " ".join(args)
    assert "-c:v h264_nvenc" in joined
    assert "-preset p7" in joined
    assert "-b:v 0" in joined
    assert "-tune hq" in joined
    assert "-multipass fullres" in joined
    assert "-spatial-aq 1" in joined
    assert "-temporal-aq 1" in joined
    assert "-rc-lookahead 20" in joined
    assert "-cq 15" in joined
    assert "-level:v 4.1" in joined
    assert "-g 60" in joined
    assert "-maxrate 14000k" in joined
    assert "-bufsize 28000k" in joined

    info = VideoInfo(
        path=Path("dummy.mov"),
        width=2160,
        height=3840,
        display_width=2160,
        display_height=3840,
        rotation=0,
        fps_num=60,
        fps_den=1,
        fps_str="60/1",
        duration=5.0,
        frame_count=300,
        has_audio=True,
        pix_fmt="yuv420p",
        codec_name="hevc",
    )
    enc_cmd = " ".join(
        build_encode_command(
            info,
            Path("out.mp4"),
            cfg,
            1080,
            1920,
            resolved_codec="h264_nvenc",
            ffmpeg_bin="ffmpeg",
        )
    )
    assert "-fps_mode cfr" in enc_cmd
    assert "-ar 48000" in enc_cmd
    assert "-level:v 4.2" in enc_cmd
    assert "-g 120" in enc_cmd
    assert "-maxrate 20000k" in enc_cmd

    auto_resolved = resolve_codec("auto")
    assert auto_resolved in (ALLOWED_CODECS - {"auto"})


def test_build_decode_command_includes_hwaccel_cuda() -> None:
    cfg = EnhanceConfig(hwaccel="cuda")
    info = VideoInfo(
        path=Path("dummy.mov"),
        width=2160,
        height=3840,
        display_width=2160,
        display_height=3840,
        rotation=0,
        fps_num=30,
        fps_den=1,
        fps_str="30/1",
        duration=5.0,
        frame_count=150,
        has_audio=True,
        pix_fmt="yuv420p",
        codec_name="hevc",
    )
    cmd = build_decode_command(
        info,
        cfg,
        target_w=1080,
        target_h=1920,
        ffmpeg_bin="ffmpeg",
    )
    hw_idx = cmd.index("-hwaccel")
    assert cmd[hw_idx + 1] == "cuda"
    assert hw_idx < cmd.index("-i")


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
    assert info.is_hdr is False
