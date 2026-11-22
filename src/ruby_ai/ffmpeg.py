"""FFmpeg and FFprobe wrappers, filter builders, and streaming pipe management."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from ruby_ai.config import EnhanceConfig


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Probed technical metadata for an input video file."""

    path: Path
    width: int
    height: int
    display_width: int
    display_height: int
    rotation: int
    fps_num: int
    fps_den: int
    fps_str: str
    duration: float
    frame_count: int
    has_audio: bool
    pix_fmt: str
    codec_name: str


def require_binary(name: str) -> str:
    """Locate an external binary on PATH or raise a helpful RuntimeError."""
    binary = shutil.which(name)
    if binary is None:
        raise RuntimeError(
            f"Required binary '{name}' was not found on PATH. "
            "Install FFmpeg (including ffprobe) and ensure it is on PATH."
        )
    return binary


def parse_display_dimensions(
    width: int,
    height: int,
    stream: dict[str, Any],
) -> tuple[int, int, int]:
    """Resolve display dimensions taking stream rotation metadata into account."""
    raw_rotation: Any = stream.get("tags", {}).get("rotate")
    if raw_rotation is None:
        for side_data in stream.get("side_data_list", []):
            if "rotation" in side_data:
                raw_rotation = side_data["rotation"]
                break

    rotation = 0
    if raw_rotation is not None:
        try:
            rotation = int(round(float(raw_rotation)))
        except (TypeError, ValueError):
            rotation = 0

    if abs(rotation) % 180 == 90:
        return height, width, rotation
    return width, height, rotation


def _parse_fps(stream: dict[str, Any]) -> tuple[int, int, str]:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key)
        if not raw or raw == "0/0":
            continue
        try:
            frac = Fraction(raw)
        except (ValueError, ZeroDivisionError):
            continue
        if frac > 0:
            return frac.numerator, frac.denominator, f"{frac.numerator}/{frac.denominator}"
    return 30, 1, "30/1"


def probe_video(path: Path) -> VideoInfo:
    """Inspect a video file using ffprobe."""
    if not path.is_file():
        raise FileNotFoundError(f"Input video not found: {path}")

    ffprobe_bin = require_binary("ffprobe")
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for '{path}': {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found in '{path}'")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    display_w, display_h, rotation = parse_display_dimensions(width, height, video_stream)
    fps_num, fps_den, fps_str = _parse_fps(video_stream)

    raw_duration = video_stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = max(0.0, float(raw_duration)) if raw_duration is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    raw_nb_frames = video_stream.get("nb_frames")
    if raw_nb_frames is not None and str(raw_nb_frames).isdigit():
        frame_count = int(raw_nb_frames)
    else:
        frame_count = max(0, round(duration * fps_num / fps_den))

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        display_width=display_w,
        display_height=display_h,
        rotation=rotation,
        fps_num=fps_num,
        fps_den=fps_den,
        fps_str=fps_str,
        duration=duration,
        frame_count=frame_count,
        has_audio=has_audio,
        pix_fmt=str(video_stream.get("pix_fmt", "unknown")),
        codec_name=str(video_stream.get("codec_name", "unknown")),
    )


def build_decode_filter(
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    src_w: int,
    src_h: int,
) -> str:
    """Build FFmpeg filtergraph for temporal pre-denoising, supersampling, and rgb24 output."""
    filters: list[str] = []
    if config.temporal_denoise > 0.0:
        a = config.temporal_denoise
        b = min(a * 2.0, 0.5)
        filters.append(
            f"atadenoise=0a={a:.4f}:0b={b:.4f}:1a={a:.4f}:1b={b:.4f}:2a={a:.4f}:2b={b:.4f}"
        )
    if (target_w, target_h) != (src_w, src_h):
        filters.append(f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd+full_chroma_int")
    filters.append("format=rgb24")
    return ",".join(filters)


def build_encode_filter(config: EnhanceConfig) -> str:
    """Build FFmpeg filtergraph for adaptive sharpening, BT.709 YUV420p conversion, and grain."""
    filters: list[str] = []
    if config.sharpen > 0.0:
        filters.append(f"cas=strength={config.sharpen:.3f}")
    filters.append("scale=out_color_matrix=bt709:out_range=tv")
    filters.append("format=yuv420p")
    if config.grain > 0:
        filters.append(f"noise=c0s={config.grain}:c0f=t+u")
    return ",".join(filters)


class ManagedProcess:
    """Subprocess wrapper draining stderr asynchronously to prevent pipe deadlocks."""

    def __init__(self, args: list[str], *, stdin_pipe: bool, stdout_pipe: bool) -> None:
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE if stdout_pipe else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw_line in self.proc.stderr:
            self._stderr_lines.append(raw_line.decode("utf-8", errors="replace").rstrip())
        self.proc.stderr.close()

    def close_stdin(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()

    def wait_and_check(self, action: str) -> str:
        code = self.proc.wait()
        self._stderr_thread.join(timeout=2.0)
        stderr_text = "\n".join(self._stderr_lines[-20:])
        if code != 0:
            raise RuntimeError(f"{action} failed with exit code {code}:\n{stderr_text}")
        return stderr_text

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._stderr_thread.join(timeout=1.0)


def _codec_args(config: EnhanceConfig) -> list[str]:
    if config.codec == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            config.x264_preset,
            "-crf",
            str(config.crf),
            "-profile:v",
            "high",
        ]
    if config.codec == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(config.crf),
            "-profile:v",
            "high",
        ]
    if config.codec == "h264_videotoolbox":
        return [
            "-c:v",
            "h264_videotoolbox",
            "-q:v",
            "65",
            "-profile:v",
            "high",
        ]
    raise ValueError(f"Unsupported codec '{config.codec}'")


def open_decode_pipe(
    info: VideoInfo,
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> ManagedProcess:
    """Open an FFmpeg process streaming decoded rgb24 frames to stdout."""
    ffmpeg_bin = require_binary("ffmpeg")
    vf = build_decode_filter(
        config,
        target_w,
        target_h,
        info.display_width,
        info.display_height,
    )
    args = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    if start_sec is not None:
        args.extend(["-ss", f"{start_sec:.3f}"])
    args.extend(["-i", str(info.path)])
    if duration_sec is not None:
        args.extend(["-t", f"{duration_sec:.3f}"])
    args.extend(["-an", "-sn", "-dn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return ManagedProcess(args, stdin_pipe=False, stdout_pipe=True)


def open_encode_pipe(
    info: VideoInfo,
    output_path: Path,
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> ManagedProcess:
    """Open an FFmpeg process reading rgb24 frames from stdin and encoding MP4."""
    ffmpeg_bin = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = build_encode_filter(config)

    args = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{target_w}x{target_h}",
        "-framerate",
        info.fps_str,
        "-i",
        "pipe:0",
    ]

    if info.has_audio:
        if start_sec is not None:
            args.extend(["-ss", f"{start_sec:.3f}"])
        if duration_sec is not None:
            args.extend(["-t", f"{duration_sec:.3f}"])
        args.extend(["-i", str(info.path)])
        args.extend(["-map", "0:v:0", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    else:
        args.extend(["-map", "0:v:0", "-an"])

    args.extend(["-vf", vf])
    args.extend(_codec_args(config))
    args.extend(
        [
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return ManagedProcess(args, stdin_pipe=True, stdout_pipe=False)


def _run_ffmpeg(args: list[str], action: str) -> None:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{action} failed with exit code {proc.returncode}:\n{proc.stderr}")


def encode_baseline_lanczos(
    input_path: Path,
    output_path: Path,
    *,
    target_w: int,
    target_h: int,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    crf: int = 16,
) -> None:
    """Encode a plain Lanczos downscaled baseline clip for A/B comparison."""
    ffmpeg_bin = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec is not None:
        args.extend(["-ss", f"{start_sec:.3f}"])
    args.extend(["-i", str(input_path)])
    if duration_sec is not None:
        args.extend(["-t", f"{duration_sec:.3f}"])
    args.extend(
        [
            "-vf",
            (
                f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
                "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-profile:v",
            "high",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    _run_ffmpeg(args, "Baseline Lanczos encode")


def simulate_social_compression(
    input_path: Path,
    output_path: Path,
    *,
    bitrate_kbps: int = 3500,
) -> None:
    """Simulate aggressive Instagram/TikTok H.264 re-encoding at a constrained bitrate."""
    ffmpeg_bin = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maxrate = int(round(bitrate_kbps * 1.15))
    bufsize = bitrate_kbps * 2
    args = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{maxrate}k",
        "-bufsize",
        f"{bufsize}k",
        "-profile:v",
        "high",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(args, "Social compression simulation")


def extract_comparison_frame(
    video_paths: list[Path],
    output_png: Path,
    *,
    timestamp_sec: float = 1.0,
) -> None:
    """Extract and horizontally stack frames from one or more videos into a PNG."""
    if not video_paths:
        raise ValueError("video_paths must not be empty")

    ffmpeg_bin = require_binary("ffmpeg")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    infos = [probe_video(p) for p in video_paths]

    positive_durations = [info.duration for info in infos if info.duration > 0.0]
    safe_ts = timestamp_sec
    if positive_durations:
        min_dur = min(positive_durations)
        safe_ts = min(timestamp_sec, max(0.0, min_dur - 0.1))

    args = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    for path in video_paths:
        args.extend(["-ss", f"{safe_ts:.3f}", "-i", str(path)])

    if len(video_paths) == 1:
        args.extend(["-frames:v", "1", str(output_png)])
    else:
        target_h = infos[0].display_height
        scale_nodes = [
            f"[{idx}:v]scale=-2:{target_h}:flags=lanczos[v{idx}]"
            for idx in range(len(video_paths))
        ]
        inputs = "".join(f"[v{idx}]" for idx in range(len(video_paths)))
        filter_complex = ";".join(scale_nodes) + f";{inputs}hstack=inputs={len(video_paths)}[out]"
        args.extend(
            ["-filter_complex", filter_complex, "-map", "[out]", "-frames:v", "1", str(output_png)]
        )

    _run_ffmpeg(args, "Comparison frame extraction")


def available_h264_encoders() -> list[str]:
    """Return supported H.264 encoders available in the installed FFmpeg binary."""
    ffmpeg_bin = require_binary("ffmpeg")
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    found: list[str] = []
    for candidate in ("libx264", "h264_nvenc", "h264_videotoolbox"):
        if candidate in proc.stdout:
            found.append(candidate)
    return found
