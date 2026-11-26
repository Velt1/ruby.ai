"""FFmpeg and FFprobe wrappers, filter builders, encoder probing, and pipe management."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from ruby_ai.config import EXPORT_TARGETS, EnhanceConfig


def subprocess_creation_flags() -> int:
    """Suppress transient console windows when spawning helper processes on Windows."""
    if platform.system() == "Windows":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x0800_0000)
    return 0


def _infer_bit_depth(pix_fmt: str) -> int:
    fmt = pix_fmt.lower()
    if "16le" in fmt or "16be" in fmt or fmt.endswith("16"):
        return 16
    if "12le" in fmt or "12be" in fmt or fmt.endswith("12"):
        return 12
    if "10le" in fmt or "10be" in fmt or "p010" in fmt or "p210" in fmt or "p410" in fmt:
        return 10
    return 8


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
    color_space: str = "unknown"
    color_transfer: str = "unknown"
    color_primaries: str = "unknown"
    color_range: str = "unknown"
    bit_depth: int = 8

    @property
    def is_hdr(self) -> bool:
        """Return True if the video uses HDR PQ or HLG transfer characteristics."""
        return self.color_transfer in {"smpte2084", "arib-std-b67"}

    @property
    def is_high_bit_depth(self) -> bool:
        """Return True for 10-bit, 12-bit, or 16-bit source pixel formats."""
        return self.bit_depth >= 10


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


def _ffprobe_json(path: Path) -> dict[str, Any]:
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
        creationflags=subprocess_creation_flags(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for '{path}': {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_video(path: Path) -> VideoInfo:
    """Inspect a video file using ffprobe."""
    payload = _ffprobe_json(path)
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

    pix_fmt = str(video_stream.get("pix_fmt", "unknown"))
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
        pix_fmt=pix_fmt,
        codec_name=str(video_stream.get("codec_name", "unknown")),
        color_space=str(video_stream.get("color_space", "unknown")),
        color_transfer=str(video_stream.get("color_transfer", "unknown")),
        color_primaries=str(video_stream.get("color_primaries", "unknown")),
        color_range=str(video_stream.get("color_range", "unknown")),
        bit_depth=_infer_bit_depth(pix_fmt),
    )


def probe_bitrate_kbps(path: Path) -> tuple[float, float]:
    """Return (total_bitrate_kbps, video_bitrate_kbps) using ffprobe."""
    payload = _ffprobe_json(path)
    raw_fmt = payload.get("format", {}).get("bit_rate")
    total_kbps = float(raw_fmt) / 1000.0 if raw_fmt and str(raw_fmt).isdigit() else 0.0
    video_stream = next(
        (s for s in payload.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    raw_vid = video_stream.get("bit_rate") if video_stream else None
    video_kbps = float(raw_vid) / 1000.0 if raw_vid and str(raw_vid).isdigit() else 0.0
    return total_kbps, video_kbps


def pipe_pix_fmt(pipe_bit_depth: int) -> str:
    """Return rawvideo pixel format corresponding to the pipeline bit depth."""
    return "rgb48le" if int(pipe_bit_depth) == 16 else "rgb24"


def _output_pix_fmt_for_codec(codec: str) -> str:
    if codec == "prores_ks":
        return "yuv422p10le"
    if codec == "hevc_nvenc":
        return "p010le"
    if codec == "libx265":
        return "yuv420p10le"
    return "yuv420p"


def build_decode_filter(
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    src_w: int,
    src_h: int,
    *,
    is_hdr: bool = False,
    color_transfer: str = "unknown",
) -> str:
    """Build FFmpeg filtergraph for optional HDR->SDR, temporal denoise, scaling, rgb24/rgb48le."""
    filters: list[str] = []
    if config.hdr_to_sdr and is_hdr:
        tin = "arib-std-b67" if color_transfer == "arib-std-b67" else "smpte2084"
        filters.append(
            f"zscale=pin=bt2020:min=bt2020nc:tin={tin}:t=linear:npl=100,"
            "format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv"
        )
    if config.temporal_denoise > 0.0:
        a = config.temporal_denoise
        b = min(a * 2.0, 0.5)
        filters.append(
            f"atadenoise=0a={a:.4f}:0b={b:.4f}:1a={a:.4f}:1b={b:.4f}:2a={a:.4f}:2b={b:.4f}"
        )
    if (target_w, target_h) != (src_w, src_h):
        filters.append(f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd+full_chroma_int")
    filters.append(f"format={pipe_pix_fmt(config.pipe_bit_depth)}")
    return ",".join(filters)


def build_encode_filter(
    config: EnhanceConfig,
    *,
    resolved_codec: str | None = None,
    input_w: int | None = None,
    input_h: int | None = None,
    target_w: int | None = None,
    target_h: int | None = None,
) -> str:
    """Build FFmpeg filtergraph for BT.709 color conversion, deband, CAS, clarity, and grain."""
    codec = resolved_codec or config.codec
    out_pix_fmt = _output_pix_fmt_for_codec(codec)

    filters: list[str] = []
    if (
        input_w is not None
        and input_h is not None
        and target_w is not None
        and target_h is not None
        and (input_w, input_h) != (target_w, target_h)
    ):
        filters.append(f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd+full_chroma_int")

    filters.extend(
        [
            "scale=out_color_matrix=bt709:out_range=tv",
            f"format={out_pix_fmt}",
        ]
    )
    if config.deband:
        filters.append("deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1")
    if config.sharpen > 0.0:
        filters.append(f"cas=strength={config.sharpen:.3f}:planes=1")
    if config.clarity > 0.0:
        la = 0.10 + config.clarity * 0.65
        contrast = 1.0 + config.clarity * 0.12
        saturation = 1.0 + config.clarity * 0.12
        filters.append(f"unsharp=lx=13:ly=13:la={la:.3f}:cx=3:cy=3:ca=0.0")
        filters.append(f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}")
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
            creationflags=subprocess_creation_flags(),
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
            try:
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def tail_stderr(self, max_lines: int = 20) -> str:
        self._stderr_thread.join(timeout=1.0)
        return "\n".join(self._stderr_lines[-max_lines:])

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


def resolve_h264_level(target_w: int, target_h: int, fps: float) -> str:
    """Select an H.264 level compatible with target dimensions and framerate."""
    if max(target_w, target_h) > 2048:
        return "5.1"
    if fps > 35.0:
        return "4.2"
    return "4.1"


def resolve_maxrate_kbps(config: EnhanceConfig, fps: float) -> int | None:
    """Resolve VBV peak bitrate cap in kbps from explicit override or export target."""
    if config.maxrate_kbps is not None:
        return config.maxrate_kbps
    target = EXPORT_TARGETS.get(config.export_target)
    if target is None:
        return None
    key = "maxrate_kbps_60fps" if fps > 35.0 else "maxrate_kbps_30fps"
    raw = target.get(key)
    return int(raw) if raw is not None else None


def resolve_gop_frames(config: EnhanceConfig, fps: float) -> int | None:
    """Compute GOP length in frames for social-ingest friendly keyframe intervals."""
    if config.gop_seconds <= 0.0:
        return None
    return max(12, round(max(1.0, fps) * config.gop_seconds))


def _build_libx264_args(
    config: EnhanceConfig,
    *,
    target_w: int,
    target_h: int,
    fps: float,
) -> list[str]:
    args = [
        "-c:v",
        "libx264",
        "-preset",
        config.x264_preset,
        "-crf",
        str(config.crf),
        "-profile:v",
        "high",
        "-level:v",
        resolve_h264_level(target_w, target_h, fps),
    ]
    gop = resolve_gop_frames(config, fps)
    if gop is not None:
        args.extend(["-g", str(gop), "-flags", "+cgop"])
    maxrate = resolve_maxrate_kbps(config, fps)
    if maxrate is not None:
        args.extend(["-maxrate", f"{maxrate}k", "-bufsize", f"{maxrate * 2}k"])
    return args


def _build_nvenc_args(
    config: EnhanceConfig,
    *,
    target_w: int,
    target_h: int,
    fps: float,
) -> list[str]:
    args = [
        "-c:v",
        "h264_nvenc",
        "-preset",
        config.nvenc_preset,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(config.crf),
        "-b:v",
        "0",
        "-multipass",
        "fullres",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-rc-lookahead",
        "20",
        "-profile:v",
        "high",
        "-level:v",
        resolve_h264_level(target_w, target_h, fps),
    ]
    gop = resolve_gop_frames(config, fps)
    if gop is not None:
        args.extend(["-g", str(gop)])
    maxrate = resolve_maxrate_kbps(config, fps)
    if maxrate is not None:
        args.extend(["-maxrate", f"{maxrate}k", "-bufsize", f"{maxrate * 2}k"])
    return args


def _build_hevc_nvenc_args(
    config: EnhanceConfig,
    *,
    target_w: int,
    target_h: int,
    fps: float,
) -> list[str]:
    del target_w, target_h
    args = [
        "-c:v",
        "hevc_nvenc",
        "-preset",
        config.nvenc_preset,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(config.crf),
        "-b:v",
        "0",
        "-multipass",
        "fullres",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-rc-lookahead",
        "20",
        "-profile:v",
        "main10",
        "-tag:v",
        "hvc1",
    ]
    gop = resolve_gop_frames(config, fps)
    if gop is not None:
        args.extend(["-g", str(gop)])
    maxrate = resolve_maxrate_kbps(config, fps)
    if maxrate is not None:
        args.extend(["-maxrate", f"{maxrate}k", "-bufsize", f"{maxrate * 2}k"])
    return args


def _build_libx265_args(
    config: EnhanceConfig,
    *,
    target_w: int,
    target_h: int,
    fps: float,
) -> list[str]:
    del target_w, target_h
    args = [
        "-c:v",
        "libx265",
        "-preset",
        config.x264_preset,
        "-crf",
        str(config.crf),
        "-profile:v",
        "main10",
        "-tag:v",
        "hvc1",
    ]
    gop = resolve_gop_frames(config, fps)
    if gop is not None:
        args.extend(["-g", str(gop)])
    maxrate = resolve_maxrate_kbps(config, fps)
    if maxrate is not None:
        args.extend(["-maxrate", f"{maxrate}k", "-bufsize", f"{maxrate * 2}k"])
    return args


def _build_prores_args(
    config: EnhanceConfig,
    *,
    target_w: int,
    target_h: int,
    fps: float,
) -> list[str]:
    del config, target_w, target_h, fps
    return [
        "-c:v",
        "prores_ks",
        "-profile:v",
        "3",
        "-vendor",
        "apl0",
    ]


def _build_videotoolbox_args(
    config: EnhanceConfig,
    *,
    target_w: int,
    target_h: int,
    fps: float,
) -> list[str]:
    args = [
        "-c:v",
        "h264_videotoolbox",
        "-q:v",
        "65",
        "-profile:v",
        "high",
        "-level:v",
        resolve_h264_level(target_w, target_h, fps),
    ]
    gop = resolve_gop_frames(config, fps)
    if gop is not None:
        args.extend(["-g", str(gop)])
    maxrate = resolve_maxrate_kbps(config, fps)
    if maxrate is not None:
        args.extend(["-maxrate", f"{maxrate}k", "-bufsize", f"{maxrate * 2}k"])
    return args


EncoderArgBuilder = Callable[..., list[str]]

ENCODER_BUILDERS: dict[str, EncoderArgBuilder] = {
    "libx264": _build_libx264_args,
    "h264_nvenc": _build_nvenc_args,
    "h264_videotoolbox": _build_videotoolbox_args,
    "hevc_nvenc": _build_hevc_nvenc_args,
    "libx265": _build_libx265_args,
    "prores_ks": _build_prores_args,
}

_SOFTWARE_CODECS = frozenset({"libx264", "libx265", "prores_ks"})


def available_video_encoders() -> list[str]:
    """Return supported video encoders listed by the installed FFmpeg binary."""
    ffmpeg_bin = require_binary("ffmpeg")
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        creationflags=subprocess_creation_flags(),
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [candidate for candidate in ENCODER_BUILDERS if candidate in proc.stdout]


def available_h264_encoders() -> list[str]:
    """Return supported H.264 encoders available in the installed FFmpeg binary."""
    h264_only = ("libx264", "h264_nvenc", "h264_videotoolbox")
    found = available_video_encoders()
    return [c for c in h264_only if c in found]


@lru_cache(maxsize=16)
def probe_encoder_usable(codec: str) -> tuple[bool, str | None]:
    """Verify whether `codec` actually initializes on the current GPU/driver at runtime."""
    if codec not in ENCODER_BUILDERS:
        return False, f"Unsupported codec '{codec}'"
    try:
        listed = available_video_encoders()
    except Exception as exc:
        return False, str(exc)
    if codec not in listed:
        return False, f"Encoder '{codec}' is not built into FFmpeg"
    if codec in _SOFTWARE_CODECS:
        return True, None

    ffmpeg_bin = require_binary("ffmpeg")
    pix_fmt = "p010le" if codec == "hevc_nvenc" else "yuv420p"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:r=30:d=0.1",
        "-frames:v",
        "1",
        "-pix_fmt",
        pix_fmt,
        "-c:v",
        codec,
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess_creation_flags(),
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    if proc.returncode == 0:
        return True, None
    err = proc.stderr.strip() or f"Encoder probe exited with {proc.returncode}"
    return False, err


def resolve_codec(requested_codec: str, *, device_backend: str | None = None) -> str:
    """Resolve explicit or 'auto' H.264 encoder based on live hardware probe."""
    if requested_codec != "auto":
        if requested_codec not in ENCODER_BUILDERS:
            raise ValueError(f"Unsupported codec '{requested_codec}'")
        return requested_codec

    encoders = available_video_encoders()
    cuda_active = device_backend == "cuda"
    if device_backend is None and "h264_nvenc" in encoders:
        try:
            import torch

            cuda_active = bool(torch.cuda.is_available())
        except Exception:
            cuda_active = False

    if cuda_active and "h264_nvenc" in encoders:
        usable, _ = probe_encoder_usable("h264_nvenc")
        if usable:
            return "h264_nvenc"
    if platform.system() == "Darwin" and "h264_videotoolbox" in encoders:
        usable, _ = probe_encoder_usable("h264_videotoolbox")
        if usable:
            return "h264_videotoolbox"
    return "libx264"


def resolve_hwaccel(requested: str, *, device_backend: str | None = None) -> str | None:
    """Resolve optional FFmpeg decode hwaccel ('none', 'auto', or 'cuda')."""
    if requested == "none":
        return None
    if requested == "cuda":
        return "cuda"
    if requested == "auto":
        if device_backend == "cuda":
            return "cuda"
        if device_backend is None:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                return None
    return None


def _codec_args(
    config: EnhanceConfig,
    *,
    resolved_codec: str | None = None,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: float = 30.0,
) -> list[str]:
    codec = resolved_codec or config.codec
    builder = ENCODER_BUILDERS.get(codec)
    if builder is None:
        raise ValueError(f"Unsupported codec '{codec}'")
    return builder(config, target_w=target_w, target_h=target_h, fps=fps)


def build_decode_command(
    info: VideoInfo,
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    hwaccel: str | None = None,
    ffmpeg_bin: str | None = None,
) -> list[str]:
    """Construct the FFmpeg command arguments used for video decoding."""
    binary = ffmpeg_bin or require_binary("ffmpeg")
    vf = build_decode_filter(
        config,
        target_w,
        target_h,
        info.display_width,
        info.display_height,
        is_hdr=info.is_hdr,
        color_transfer=info.color_transfer,
    )
    args = [binary, "-hide_banner", "-loglevel", "error"]
    resolved_hw = hwaccel if hwaccel is not None else resolve_hwaccel(config.hwaccel)
    if resolved_hw:
        args.extend(["-hwaccel", resolved_hw])
    if start_sec is not None:
        args.extend(["-ss", f"{start_sec:.3f}"])
    args.extend(["-i", str(info.path)])
    if duration_sec is not None:
        args.extend(["-t", f"{duration_sec:.3f}"])
    args.extend(
        [
            "-an",
            "-sn",
            "-dn",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            pipe_pix_fmt(config.pipe_bit_depth),
            "pipe:1",
        ]
    )
    return args


def open_decode_pipe(
    info: VideoInfo,
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    hwaccel: str | None = None,
) -> ManagedProcess:
    """Open an FFmpeg process streaming decoded rgb24/rgb48le frames to stdout."""
    args = build_decode_command(
        info,
        config,
        target_w,
        target_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
        hwaccel=hwaccel,
    )
    return ManagedProcess(args, stdin_pipe=False, stdout_pipe=True)


def build_encode_command(
    info: VideoInfo,
    output_path: Path,
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    *,
    input_w: int | None = None,
    input_h: int | None = None,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    resolved_codec: str | None = None,
    ffmpeg_bin: str | None = None,
) -> list[str]:
    """Construct the FFmpeg command arguments used for video encoding."""
    binary = ffmpeg_bin or require_binary("ffmpeg")
    in_w = input_w or target_w
    in_h = input_h or target_h
    codec = resolved_codec or config.codec
    vf = build_encode_filter(
        config,
        resolved_codec=codec,
        input_w=in_w,
        input_h=in_h,
        target_w=target_w,
        target_h=target_h,
    )
    fps_val = info.fps_num / info.fps_den if info.fps_den else 30.0

    args = [
        binary,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pipe_pix_fmt(config.pipe_bit_depth),
        "-s:v",
        f"{in_w}x{in_h}",
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
        args.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(config.audio_sample_rate),
                "-shortest",
            ]
        )
    else:
        args.extend(["-map", "0:v:0", "-an"])

    args.extend(["-vf", vf])
    if config.cfr:
        args.extend(["-fps_mode", "cfr"])
    args.extend(
        _codec_args(
            config,
            resolved_codec=codec,
            target_w=target_w,
            target_h=target_h,
            fps=fps_val,
        )
    )
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
        ]
    )
    if output_path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        args.extend(["-movflags", "+faststart"])
    args.append(str(output_path))
    return args


def open_encode_pipe(
    info: VideoInfo,
    output_path: Path,
    config: EnhanceConfig,
    target_w: int,
    target_h: int,
    *,
    input_w: int | None = None,
    input_h: int | None = None,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    resolved_codec: str | None = None,
) -> ManagedProcess:
    """Open an FFmpeg process reading rgb24/rgb48le frames from stdin and encoding video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = build_encode_command(
        info,
        output_path,
        config,
        target_w,
        target_h,
        input_w=input_w,
        input_h=input_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
        resolved_codec=resolved_codec,
    )
    return ManagedProcess(args, stdin_pipe=True, stdout_pipe=False)


def _run_ffmpeg(args: list[str], action: str) -> None:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        creationflags=subprocess_creation_flags(),
        check=False,
    )
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
    fast_preview: bool = False,
    hdr_to_sdr: bool = False,
    is_hdr: bool = False,
    color_transfer: str = "unknown",
) -> None:
    """Encode a plain Lanczos baseline clip for A/B comparison."""
    ffmpeg_bin = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_prefix = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if start_sec is not None:
        base_prefix.extend(["-ss", f"{start_sec:.3f}"])
    base_prefix.extend(["-i", str(input_path)])
    if duration_sec is not None:
        base_prefix.extend(["-t", f"{duration_sec:.3f}"])

    filters: list[str] = []
    if hdr_to_sdr and is_hdr:
        tin = "arib-std-b67" if color_transfer == "arib-std-b67" else "smpte2084"
        filters.append(
            f"zscale=pin=bt2020:min=bt2020nc:tin={tin}:t=linear:npl=100,"
            "format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv"
        )
    filters.append(
        f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
    )
    vf = ",".join(filters)

    color_and_container = [
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
    ]

    if fast_preview:
        encoder_candidates: list[list[str]] = []
        if probe_encoder_usable("h264_nvenc")[0]:
            encoder_candidates.append(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    "18",
                    "-b:v",
                    "0",
                    "-profile:v",
                    "high",
                ]
            )
        encoder_candidates.append(
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-profile:v", "high"]
        )
        middle_args = ["-vf", vf, "-fps_mode", "cfr", "-g", "30", "-an"]
    else:
        encoder_candidates = [
            ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high"]
        ]
        middle_args = ["-vf", vf, "-c:a", "aac", "-b:a", "192k"]

    for idx, encoder_args in enumerate(encoder_candidates):
        cmd = [
            *base_prefix,
            *middle_args,
            *encoder_args,
            *color_and_container,
            str(output_path),
        ]
        is_last = idx == len(encoder_candidates) - 1
        try:
            _run_ffmpeg(cmd, "Baseline Lanczos encode")
            return
        except Exception:
            if is_last:
                raise


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
