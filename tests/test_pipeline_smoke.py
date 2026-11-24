"""End-to-end smoke tests for FFmpeg streaming enhance and benchmark pipelines."""

import subprocess
from pathlib import Path

from ruby_ai.config import EnhanceConfig
from ruby_ai.ffmpeg import probe_video, require_binary
from ruby_ai.pipeline import BenchmarkReport, enhance_video, run_benchmark_detailed


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


def test_enhance_video_sync_and_async_plus_benchmark_report(tmp_path: Path) -> None:
    src = tmp_path / "input.mp4"
    _make_synthetic_clip(src)

    progress_events: list[tuple[int, int | None]] = []
    cfg_async = EnhanceConfig.from_preset(
        "passthrough",
        max_short_edge=160,
        x264_preset="ultrafast",
        batch_size=2,
        async_io=True,
    )

    out_async = tmp_path / "out_async.mp4"
    result_async = enhance_video(
        src,
        out_async,
        cfg_async,
        show_progress=False,
        on_progress=lambda done, total: progress_events.append((done, total)),
    )
    assert out_async.is_file()
    assert result_async.target_width == 160
    assert result_async.target_height == 240
    assert result_async.frames_processed >= 10
    assert result_async.codec_used == "libx264"
    assert result_async.batch_size == 2
    assert progress_events
    assert progress_events[-1][0] == result_async.frames_processed

    cfg_sync = EnhanceConfig.from_preset(
        "passthrough",
        max_short_edge=160,
        x264_preset="ultrafast",
        batch_size=2,
        async_io=False,
    )
    out_sync = tmp_path / "out_sync.mp4"
    result_sync = enhance_video(src, out_sync, cfg_sync, show_progress=False)
    assert result_sync.frames_processed == result_async.frames_processed

    probed = probe_video(out_async)
    assert probed.display_width == 160
    assert probed.display_height == 240
    assert probed.codec_name == "h264"
    assert probed.pix_fmt == "yuv420p"
    assert probed.has_audio is True

    bench_dir = tmp_path / "bench"
    report = run_benchmark_detailed(
        src,
        bench_dir,
        cfg_async,
        duration_sec=0.5,
        bitrate_kbps=1500,
        show_progress=False,
    )
    assert isinstance(report, BenchmarkReport)
    assert report.enhance_result.fps > 0
    assert report.artifacts["baseline"].is_file()
    assert report.artifacts["enhanced"].is_file()
    assert report.artifacts["baseline_social"].is_file()
    assert report.artifacts["enhanced_social"].is_file()
    assert report.artifacts["comparison_png"].is_file()
    assert report.artifacts["comparison_png"].stat().st_size > 0
