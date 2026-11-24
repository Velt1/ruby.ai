"""Tests for the NDJSON desktop bridge module."""

import io
import json
import subprocess
import sys
from pathlib import Path

from ruby_ai.bridge import BRIDGE_COMMANDS, BridgeEmitter, run_bridge_command
from ruby_ai.config import EXPORT_TARGETS, PRESETS
from ruby_ai.ffmpeg import probe_video, require_binary
from ruby_ai.models import MODEL_REGISTRY


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
            "testsrc2=size=320x480:rate=12:duration=0.8",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=0.8",
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


def _parse_ndjson(raw: str) -> list[dict]:
    return [json.loads(line) for line in raw.strip().splitlines() if line.strip()]


def test_bridge_info_in_process_and_subprocess() -> None:
    assert {"info", "probe", "analyze", "enhance", "benchmark", "preview"}.issubset(
        BRIDGE_COMMANDS.keys()
    )

    buf = io.StringIO()
    code = run_bridge_command("info", {}, emitter=BridgeEmitter(buf))
    assert code == 0

    events = _parse_ndjson(buf.getvalue())
    assert len(events) == 1
    assert events[0]["type"] == "result"
    data = events[0]["data"]
    assert {m["key"] for m in data["models"]} == set(MODEL_REGISTRY.keys())
    assert {p["key"] for p in data["presets"]} == set(PRESETS.keys())
    assert {t["key"] for t in data["export_targets"]} == set(EXPORT_TARGETS.keys())
    assert len(data["profiles"]) >= 4
    assert "codecs" in data["options"]

    proc = subprocess.run(
        [sys.executable, "-m", "ruby_ai.bridge", "info"],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    sub_events = _parse_ndjson(proc.stdout)
    assert len(sub_events) == 1
    assert sub_events[0]["type"] == "result"


def test_bridge_probe_analyze_preview_and_enhance_ndjson(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RUBY_AI_CACHE_DIR", str(tmp_path / "cache"))
    src = tmp_path / "input.mp4"
    _make_synthetic_clip(src)

    probe_buf = io.StringIO()
    assert (
        run_bridge_command(
            "probe",
            {"path": str(src), "max_short_edge": 160},
            emitter=BridgeEmitter(probe_buf),
        )
        == 0
    )
    probe_events = _parse_ndjson(probe_buf.getvalue())
    assert probe_events[0]["type"] == "result"
    probe_data = probe_events[0]["data"]
    assert probe_data["target_width"] == 160
    assert probe_data["target_height"] == 240
    assert probe_data["poster_path"] and Path(probe_data["poster_path"]).is_file()

    analyze_buf = io.StringIO()
    assert (
        run_bridge_command(
            "analyze",
            {"path": str(src)},
            emitter=BridgeEmitter(analyze_buf),
        )
        == 0
    )
    analyze_events = _parse_ndjson(analyze_buf.getvalue())
    assert analyze_events[0]["type"] == "result"
    analyze_data = analyze_events[0]["data"]
    assert analyze_data["recommended_preset"] in PRESETS
    assert analyze_data["recommended_export_target"] == "social-1080p"
    assert len(analyze_data["reasons"]) >= 1

    preview_buf = io.StringIO()
    assert (
        run_bridge_command(
            "preview",
            {
                "job_id": "job-prev-1",
                "input_path": str(src),
                "preset": "passthrough",
                "duration_sec": 0.5,
                "bitrate_kbps": 1500,
                "config": {"max_short_edge": 160, "x264_preset": "ultrafast"},
            },
            emitter=BridgeEmitter(preview_buf),
        )
        == 0
    )
    prev_events = _parse_ndjson(preview_buf.getvalue())
    assert prev_events[0]["type"] == "started"
    assert prev_events[-1]["type"] == "completed"
    assert prev_events[-1]["mode"] == "preview"
    prev_res = prev_events[-1]["result"]
    assert Path(prev_res["before_frame_png"]).is_file()
    assert Path(prev_res["after_frame_png"]).is_file()

    enh_out = tmp_path / "enhanced.mp4"
    enh_buf = io.StringIO()
    assert (
        run_bridge_command(
            "enhance",
            {
                "job_id": "job-enh-1",
                "input_path": str(src),
                "output_path": str(enh_out),
                "preset": "passthrough",
                "config": {"max_short_edge": 160, "x264_preset": "ultrafast"},
            },
            emitter=BridgeEmitter(enh_buf),
        )
        == 0
    )
    enh_events = _parse_ndjson(enh_buf.getvalue())
    assert enh_events[0]["type"] == "started"
    assert any(e["type"] == "progress" for e in enh_events)
    assert enh_events[-1]["type"] == "completed"
    enh_res = enh_events[-1]["result"]
    assert Path(enh_res["output_path"]).is_file()
    assert Path(enh_res["before_frame_png"]).is_file()
    assert Path(enh_res["after_frame_png"]).is_file()
    before_video = Path(enh_res["before_video_path"])
    assert before_video.is_file()
    before_probe = probe_video(before_video)
    assert before_probe.codec_name == "h264"
    assert (before_probe.display_width, before_probe.display_height) == (160, 240)
