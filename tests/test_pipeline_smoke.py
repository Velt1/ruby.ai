"""End-to-end smoke tests for FFmpeg streaming enhance and benchmark pipelines."""

import subprocess
from pathlib import Path

from ruby_ai.config import EnhanceConfig
from ruby_ai.ffmpeg import probe_video, require_binary
from ruby_ai.pipeline import enhance_video, run_benchmark


def _make_synthetic_clip(path: Path) -> None:
    ffmpeg_bin = require_binary("ffmpeg")
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x480:rate=12:duration=1.0",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=1.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def test_enhance_video_passthrough_and_benchmark(tmp_path: Path) -> None:
    src = tmp_path / "input.mp4"
    _make_synthetic_clip(src)

    cfg = EnhanceConfig.from_preset(
        "passthrough",
        max_short_edge=160,
        x264_preset="ultrafast",
    )

    out = tmp_path / "out.mp4"
    result = enhance_video(src, out, cfg, show_progress=False)
    assert out.is_file()
    assert result.target_width == 160
    assert result.target_height == 240
    assert result.frames_processed >= 10

    probed = probe_video(out)
    assert probed.display_width == 160
    assert probed.display_height == 240
    assert probed.codec_name == "h264"
    assert probed.pix_fmt == "yuv420p"
    assert probed.has_audio is True

    bench_dir = tmp_path / "bench"
    artifacts = run_benchmark(
        src,
        bench_dir,
        cfg,
        duration_sec=0.5,
        bitrate_kbps=1500,
        show_progress=False,
    )
    assert artifacts["baseline"].is_file()
    assert artifacts["enhanced"].is_file()
    assert artifacts["baseline_social"].is_file()
    assert artifacts["enhanced_social"].is_file()
    assert artifacts["comparison_png"].is_file()
    assert artifacts["comparison_png"].stat().st_size > 0
