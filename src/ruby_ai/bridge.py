"""Extensible NDJSON CLI bridge connecting desktop frontends to the ruby.ai engine."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, TextIO, cast

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Keep module-level imports free of PyTorch/Spandrel so `probe` & `analyze` stay instant.
from ruby_ai import __version__
from ruby_ai.analyzer import analyze_video
from ruby_ai.config import (
    ALLOWED_CODECS,
    ALLOWED_EXPORT_TARGETS,
    ALLOWED_HWACCELS,
    ALLOWED_NVENC_PRESETS,
    ALLOWED_PIPE_BIT_DEPTHS,
    ALLOWED_TILE_BLENDINGS,
    ALLOWED_X264_PRESETS,
    EXPORT_TARGETS,
    PRESETS,
    PROFILE_METADATA,
    EnhanceConfig,
    PresetName,
    resolve_target_dimensions,
)
from ruby_ai.ffmpeg import (
    available_video_encoders,
    encode_baseline_lanczos,
    probe_bitrate_kbps,
    probe_encoder_usable,
    probe_video,
    require_binary,
    subprocess_creation_flags,
)

_KNOWN_UPSCALE_SCALES: dict[str, int] = {
    "upscale-2x-span": 2,
}


def _preview_cache_dir() -> Path:
    env_dir = os.environ.get("RUBY_AI_CACHE_DIR")
    if env_dir:
        base_dir = Path(env_dir).expanduser().parent
    else:
        base_dir = Path.home() / ".cache" / "ruby-ai"
    cache_dir = base_dir / "previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class BridgeEmitter:
    """Thread-safe NDJSON event emitter writing one JSON line per event."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        reconfigure = getattr(self._stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", line_buffering=True)
            except Exception:
                pass

    def emit(self, event_type: str, **payload: Any) -> None:
        record = {"type": event_type, **payload}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


BridgeCommandHandler = Callable[[dict[str, Any], BridgeEmitter], None]


def _extract_single_frame(
    video_path: Path,
    output_png: Path,
    *,
    timestamp_sec: float = 0.5,
    max_height: int | None = 720,
    scale_w: int | None = None,
    scale_h: int | None = None,
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)
    safe_ts = (
        min(max(0.0, timestamp_sec), max(0.0, info.duration - 0.1))
        if info.duration > 0.0
        else 0.0
    )

    args = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{safe_ts:.3f}",
        "-i",
        str(video_path),
    ]
    if scale_w is not None and scale_h is not None:
        args.extend(
            [
                "-vf",
                f"scale={scale_w}:{scale_h}:flags=lanczos+accurate_rnd+full_chroma_int",
            ]
        )
    elif max_height is not None:
        args.extend(["-vf", f"scale=-2:'min(ih,{max_height})':flags=lanczos"])
    args.extend(["-frames:v", "1", str(output_png)])

    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        creationflags=subprocess_creation_flags(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Poster/frame extraction failed for '{video_path}': {proc.stderr.strip()}"
        )
    return output_png


def _build_enhance_config(payload: dict[str, Any]) -> tuple[PresetName, EnhanceConfig]:
    preset_raw = str(payload.get("preset") or "social-crisp")
    if preset_raw not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_raw}'. Expected one of: {', '.join(sorted(PRESETS))}"
        )
    preset = cast(PresetName, preset_raw)

    raw_overrides = payload.get("config") or {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("config must be a JSON object")

    valid_fields = {f.name for f in fields(EnhanceConfig)}
    clean_overrides = {
        k: v for k, v in raw_overrides.items() if k in valid_fields and v is not None
    }

    export_target = clean_overrides.get("export_target")
    if (
        isinstance(export_target, str)
        and export_target in EXPORT_TARGETS
        and "max_short_edge" not in clean_overrides
    ):
        clean_overrides["max_short_edge"] = EXPORT_TARGETS[export_target]["max_short_edge"]

    return preset, EnhanceConfig.from_preset(preset, **clean_overrides)


def _handle_info(payload: dict[str, Any], emitter: BridgeEmitter) -> None:
    del payload
    import spandrel
    import torch

    from ruby_ai.device import (
        detect_nvidia_gpu_hardware,
        query_cuda_memory_info,
        select_device,
    )
    from ruby_ai.models import MODEL_REGISTRY

    device_payload: dict[str, Any]
    free_memory_gb: float | None = None
    try:
        dev = select_device("auto")
        mem_info = query_cuda_memory_info(dev)
        if mem_info is not None:
            free_memory_gb = round(mem_info[0] / (1024**3), 2)
        device_payload = {
            "backend": dev.backend,
            "name": dev.name,
            "default_fp16": dev.default_fp16,
            "total_memory_gb": (
                round(dev.total_memory_bytes / (1024**3), 2)
                if dev.total_memory_bytes is not None
                else None
            ),
            "free_memory_gb": free_memory_gb,
            "error": None,
        }
    except Exception as exc:
        device_payload = {
            "backend": "cpu",
            "name": "Unavailable",
            "default_fp16": False,
            "total_memory_gb": None,
            "free_memory_gb": None,
            "error": str(exc),
        }

    cuda_avail = bool(torch.cuda.is_available())
    nv_hw = detect_nvidia_gpu_hardware()
    cuda_warning: str | None = None
    if not cuda_avail and nv_hw is not None:
        cuda_warning = (
            f"NVIDIA GPU erkannt ({nv_hw['name']}, Treiber {nv_hw['driver_version']}), "
            "aber CUDA ist in PyTorch inaktiv. Bitte NVIDIA-Treiber (>= 560) aktualisieren."
        )

    mps_backend = getattr(torch.backends, "mps", None)
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    try:
        encoders = available_video_encoders() if ffmpeg_path else []
    except Exception:
        encoders = []

    nvenc_usable: bool | None = None
    nvenc_error: str | None = None
    if ffmpeg_path and "h264_nvenc" in encoders and (cuda_avail or nv_hw is not None):
        nvenc_usable, nvenc_error = probe_encoder_usable("h264_nvenc")

    presets_payload = [{"key": key, **values} for key, values in PRESETS.items()]
    export_targets_payload = [{"key": key, **values} for key, values in EXPORT_TARGETS.items()]
    models_payload = [
        {
            "key": spec.key,
            "architecture": spec.architecture,
            "scale": spec.scale,
            "author": spec.author,
            "license": spec.license,
            "description": spec.description,
            "source_url": spec.source_url,
            "source_kind": spec.source.source_kind,
            "commercial_use": spec.commercial_use,
        }
        for spec in MODEL_REGISTRY.values()
    ]

    emitter.emit(
        "result",
        data={
            "version": __version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "spandrel": getattr(spandrel, "__version__", "installed"),
            "device": device_payload,
            "cuda_available": cuda_avail,
            "mps_available": bool(mps_backend and mps_backend.is_available()),
            "nvidia_gpu_hardware": nv_hw["name"] if nv_hw else None,
            "nvidia_driver_version": nv_hw["driver_version"] if nv_hw else None,
            "cuda_warning": cuda_warning,
            "ffmpeg_available": ffmpeg_path is not None,
            "ffprobe_available": ffprobe_path is not None,
            "ffmpeg_path": ffmpeg_path,
            "ffprobe_path": ffprobe_path,
            "encoders": encoders,
            "nvenc_usable": nvenc_usable,
            "nvenc_error": nvenc_error,
            "presets": presets_payload,
            "profiles": PROFILE_METADATA,
            "export_targets": export_targets_payload,
            "models": models_payload,
            "options": {
                "codecs": sorted(ALLOWED_CODECS),
                "x264_presets": sorted(ALLOWED_X264_PRESETS),
                "nvenc_presets": sorted(ALLOWED_NVENC_PRESETS),
                "hwaccels": sorted(ALLOWED_HWACCELS),
                "export_targets": sorted(ALLOWED_EXPORT_TARGETS),
                "tile_blendings": sorted(ALLOWED_TILE_BLENDINGS),
                "pipe_bit_depths": sorted(ALLOWED_PIPE_BIT_DEPTHS),
            },
        },
    )


def _handle_probe(payload: dict[str, Any], emitter: BridgeEmitter) -> None:
    raw_path = payload.get("path")
    if not raw_path:
        raise ValueError("probe requires a non-empty 'path'")

    video_path = Path(str(raw_path)).expanduser()
    info = probe_video(video_path)
    max_short_edge = int(payload.get("max_short_edge") or 1080)
    model_key = str(payload.get("model") or "")
    model_scale = int(payload.get("model_scale") or _KNOWN_UPSCALE_SCALES.get(model_key, 1))
    target_w, target_h = resolve_target_dimensions(
        info.display_width,
        info.display_height,
        max_short_edge,
        allow_upscale=model_scale > 1,
        model_scale=model_scale,
    )
    stat = video_path.stat()
    try:
        total_kbps, video_kbps = probe_bitrate_kbps(video_path)
    except Exception:
        total_kbps, video_kbps = 0.0, 0.0

    poster_path: str | None = None
    if payload.get("generate_poster", True):
        key = hashlib.sha256(
            f"{video_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()
        ).hexdigest()[:16]
        poster_file = _preview_cache_dir() / f"poster_{key}.png"
        if not poster_file.is_file() or poster_file.stat().st_size == 0:
            _extract_single_frame(
                video_path,
                poster_file,
                timestamp_sec=min(1.0, max(0.0, info.duration * 0.15)),
                max_height=720,
            )
        poster_path = str(poster_file)

    emitter.emit(
        "result",
        data={
            "path": str(video_path),
            "filename": video_path.name,
            "width": info.width,
            "height": info.height,
            "display_width": info.display_width,
            "display_height": info.display_height,
            "rotation": info.rotation,
            "fps_num": info.fps_num,
            "fps_den": info.fps_den,
            "fps": round(info.fps_num / info.fps_den, 3) if info.fps_den else 30.0,
            "fps_str": info.fps_str,
            "duration": info.duration,
            "frame_count": info.frame_count,
            "has_audio": info.has_audio,
            "pix_fmt": info.pix_fmt,
            "bit_depth": info.bit_depth,
            "is_high_bit_depth": info.is_high_bit_depth,
            "codec_name": info.codec_name,
            "color_space": info.color_space,
            "color_transfer": info.color_transfer,
            "is_hdr": info.is_hdr,
            "file_size_bytes": stat.st_size,
            "total_bitrate_kbps": round(total_kbps, 1),
            "video_bitrate_kbps": round(video_kbps, 1),
            "target_width": target_w,
            "target_height": target_h,
            "poster_path": poster_path,
        },
    )


def _handle_analyze(payload: dict[str, Any], emitter: BridgeEmitter) -> None:
    raw_path = payload.get("path")
    if not raw_path:
        raise ValueError("analyze requires a non-empty 'path'")

    analysis = analyze_video(Path(str(raw_path)).expanduser())
    emitter.emit(
        "result",
        data={
            "path": str(analysis.path),
            "avg_luma": analysis.avg_luma,
            "avg_saturation": analysis.avg_saturation,
            "bits_per_pixel": analysis.bits_per_pixel,
            "noise_score": analysis.noise_score,
            "sharpness_score": analysis.sharpness_score,
            "compression_score": analysis.compression_score,
            "is_low_light": analysis.is_low_light,
            "is_heavily_compressed": analysis.is_heavily_compressed,
            "is_hdr": analysis.is_hdr,
            "is_high_fps": analysis.is_high_fps,
            "recommended_preset": analysis.recommended_preset,
            "recommended_export_target": analysis.recommended_export_target,
            "recommended_overrides": analysis.recommended_overrides,
            "reasons": analysis.reasons,
        },
    )


def _make_progress_callback(
    emitter: BridgeEmitter,
    *,
    job_id: str,
    stage: str,
) -> Callable[[int, int | None], None]:
    started = time.perf_counter()
    last_emit = 0.0

    def _on_progress(frames_done: int, total_frames: int | None) -> None:
        nonlocal last_emit
        now = time.perf_counter()
        is_final = total_frames is not None and frames_done >= total_frames
        if not is_final and (now - last_emit) < 0.15:
            return
        last_emit = now
        elapsed = max(1e-6, now - started)
        fps = frames_done / elapsed
        percent = (
            min(100.0, (frames_done / total_frames) * 100.0)
            if total_frames and total_frames > 0
            else None
        )
        eta_sec = (
            max(0.0, (total_frames - frames_done) / fps)
            if total_frames and total_frames > frames_done and fps > 0.0
            else None
        )
        emitter.emit(
            "progress",
            job_id=job_id,
            stage=stage,
            frames_done=frames_done,
            total_frames=total_frames,
            percent=round(percent, 1) if percent is not None else None,
            fps=round(fps, 2),
            elapsed_sec=round(elapsed, 2),
            eta_sec=round(eta_sec, 1) if eta_sec is not None else None,
        )

    return _on_progress


def _make_status_callback(
    emitter: BridgeEmitter,
    *,
    job_id: str,
) -> Callable[[str, str, str | None], None]:
    def _on_status(stage: str, message: str, detail: str | None) -> None:
        emitter.emit(
            "status",
            job_id=job_id,
            stage=stage,
            message=message,
            detail=detail,
        )

    return _on_status


def _handle_enhance(payload: dict[str, Any], emitter: BridgeEmitter) -> None:
    from ruby_ai.pipeline import enhance_video

    job_id = str(payload.get("job_id") or "job")
    raw_input = payload.get("input_path")
    if not raw_input:
        raise ValueError("enhance requires 'input_path'")

    input_path = Path(str(raw_input)).expanduser()
    preset, config = _build_enhance_config(payload)
    ext = ".mov" if config.codec == "prores_ks" else ".mp4"
    raw_output = payload.get("output_path")
    output_path = (
        Path(str(raw_output)).expanduser()
        if raw_output
        else input_path.with_name(f"{input_path.stem}_ruby_{preset}{ext}")
    )
    start_sec = float(payload["start_sec"]) if payload.get("start_sec") is not None else None
    duration_sec = (
        float(payload["duration_sec"]) if payload.get("duration_sec") is not None else None
    )

    emitter.emit(
        "started",
        job_id=job_id,
        mode="enhance",
        input_path=str(input_path),
        output_path=str(output_path),
        preset=preset,
        model=config.model,
    )

    status_cb = _make_status_callback(emitter, job_id=job_id)
    result = enhance_video(
        input_path,
        output_path,
        config,
        start_sec=start_sec,
        duration_sec=duration_sec,
        show_progress=False,
        on_progress=_make_progress_callback(emitter, job_id=job_id, stage="enhance"),
        on_status=status_cb,
    )

    before_png: str | None = None
    after_png: str | None = None
    before_video_path: str | None = None
    if payload.get("extract_preview_pair", True):
        status_cb("preview_pair", "Erstelle Vorher/Nachher-Vergleichsbilder…", None)
        try:
            src_info = probe_video(input_path)
            out_info = probe_video(result.output_path)
            mid_out = max(0.05, out_info.duration * 0.5) if out_info.duration > 0 else 0.0
            mid_in = (start_sec or 0.0) + mid_out
            preview_dir = _preview_cache_dir() / job_id
            before_file = preview_dir / "before_frame.png"
            after_file = preview_dir / "after_frame.png"
            _extract_single_frame(
                input_path,
                before_file,
                timestamp_sec=mid_in,
                max_height=None,
                scale_w=result.target_width,
                scale_h=result.target_height,
            )
            _extract_single_frame(
                result.output_path,
                after_file,
                timestamp_sec=mid_out,
                max_height=None,
            )
            before_png = str(before_file)
            after_png = str(after_file)

            # Avoid re-encoding full-length comparison baselines on long exports (>45s).
            if out_info.duration <= 45.0:
                before_video_file = preview_dir / "before_video.mp4"
                encode_baseline_lanczos(
                    input_path,
                    before_video_file,
                    target_w=result.target_width,
                    target_h=result.target_height,
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    fast_preview=True,
                    hdr_to_sdr=config.hdr_to_sdr,
                    is_hdr=src_info.is_hdr,
                    color_transfer=src_info.color_transfer,
                )
                before_video_path = str(before_video_file)
        except Exception:
            # Never fail an already-completed export due to optional preview extraction.
            pass

    file_size_bytes = result.output_path.stat().st_size if result.output_path.is_file() else 0
    try:
        total_kbps, video_kbps = probe_bitrate_kbps(result.output_path)
    except Exception:
        total_kbps, video_kbps = 0.0, 0.0

    emitter.emit(
        "completed",
        job_id=job_id,
        mode="enhance",
        result={
            "output_path": str(result.output_path),
            "frames_processed": result.frames_processed,
            "elapsed_sec": round(result.elapsed_sec, 2),
            "fps": round(result.fps, 2),
            "device_label": result.device_label,
            "target_width": result.target_width,
            "target_height": result.target_height,
            "model_key": result.model_key,
            "codec_used": result.codec_used,
            "effective_tile_size": result.effective_tile_size,
            "effective_tile_batch_size": result.effective_tile_batch_size,
            "pipe_bit_depth": result.pipe_bit_depth,
            "model_scale": result.model_scale,
            "batch_size": result.batch_size,
            "file_size_bytes": file_size_bytes,
            "total_bitrate_kbps": round(total_kbps, 1),
            "video_bitrate_kbps": round(video_kbps, 1),
            "before_frame_png": before_png,
            "after_frame_png": after_png,
            "before_video_path": before_video_path,
        },
    )


def _run_comparison_job(
    payload: dict[str, Any],
    emitter: BridgeEmitter,
    *,
    mode: str,
    default_duration: float,
    default_bitrate_kbps: int,
    default_output_dir: Path,
    default_start_sec: float | None = None,
    fast_baseline: bool = False,
) -> None:
    from ruby_ai.pipeline import run_benchmark_detailed

    job_id = str(payload.get("job_id") or "job")
    raw_input = payload.get("input_path")
    if not raw_input:
        raise ValueError(f"{mode} requires 'input_path'")

    input_path = Path(str(raw_input)).expanduser()
    preset, config = _build_enhance_config(payload)
    raw_dir = payload.get("output_dir")
    output_dir = Path(str(raw_dir)).expanduser() if raw_dir else default_output_dir
    start_sec = (
        float(payload["start_sec"]) if payload.get("start_sec") is not None else default_start_sec
    )
    duration_sec = float(payload.get("duration_sec") or default_duration)
    bitrate_kbps = int(payload.get("bitrate_kbps") or default_bitrate_kbps)
    raw_ref = payload.get("reference_path")
    reference_path = Path(str(raw_ref)).expanduser() if raw_ref else None

    emitter.emit(
        "started",
        job_id=job_id,
        mode=mode,
        input_path=str(input_path),
        output_dir=str(output_dir),
        preset=preset,
        model=config.model,
    )

    status_cb = _make_status_callback(emitter, job_id=job_id)
    report = run_benchmark_detailed(
        input_path,
        output_dir,
        config,
        start_sec=start_sec,
        duration_sec=duration_sec,
        bitrate_kbps=bitrate_kbps,
        reference_path=reference_path,
        show_progress=False,
        on_progress=_make_progress_callback(emitter, job_id=job_id, stage=mode),
        on_status=status_cb,
        fast_baseline=fast_baseline,
    )

    compare_ts = max(0.1, duration_sec * 0.5)
    before_file = output_dir / "before_frame.png"
    after_file = output_dir / "after_frame.png"
    _extract_single_frame(
        report.artifacts["baseline_social"],
        before_file,
        timestamp_sec=compare_ts,
        max_height=None,
    )
    _extract_single_frame(
        report.artifacts["enhanced_social"],
        after_file,
        timestamp_sec=compare_ts,
        max_height=None,
    )

    artifacts_payload: dict[str, dict[str, Any]] = {}
    for key, artifact_path in report.artifacts.items():
        size_bytes = artifact_path.stat().st_size if artifact_path.is_file() else 0
        if artifact_path.suffix.lower() == ".mp4" and artifact_path.is_file():
            tot_kbps, vid_kbps = probe_bitrate_kbps(artifact_path)
        else:
            tot_kbps, vid_kbps = 0.0, 0.0
        artifacts_payload[key] = {
            "path": str(artifact_path),
            "size_bytes": size_bytes,
            "total_bitrate_kbps": round(tot_kbps, 1),
            "video_bitrate_kbps": round(vid_kbps, 1),
        }

    res = report.enhance_result
    emitter.emit(
        "completed",
        job_id=job_id,
        mode=mode,
        result={
            "output_dir": str(output_dir),
            "bitrate_kbps": report.bitrate_kbps,
            "artifacts": artifacts_payload,
            "before_frame_png": str(before_file),
            "after_frame_png": str(after_file),
            "enhance_result": {
                "output_path": str(res.output_path),
                "frames_processed": res.frames_processed,
                "elapsed_sec": round(res.elapsed_sec, 2),
                "fps": round(res.fps, 2),
                "device_label": res.device_label,
                "target_width": res.target_width,
                "target_height": res.target_height,
                "model_key": res.model_key,
                "codec_used": res.codec_used,
                "effective_tile_size": res.effective_tile_size,
                "effective_tile_batch_size": res.effective_tile_batch_size,
                "pipe_bit_depth": res.pipe_bit_depth,
                "model_scale": res.model_scale,
                "batch_size": res.batch_size,
            },
        },
    )


def _handle_benchmark(payload: dict[str, Any], emitter: BridgeEmitter) -> None:
    job_id = str(payload.get("job_id") or "job")
    raw_input = payload.get("input_path")
    if not raw_input:
        raise ValueError("benchmark requires 'input_path'")
    input_path = Path(str(raw_input)).expanduser()
    _run_comparison_job(
        payload,
        emitter,
        mode="benchmark",
        default_duration=5.0,
        default_bitrate_kbps=3500,
        default_output_dir=input_path.parent / f"{input_path.stem}_ruby_benchmark_{job_id[:8]}",
        fast_baseline=False,
    )


def _handle_preview(payload: dict[str, Any], emitter: BridgeEmitter) -> None:
    job_id = str(payload.get("job_id") or "preview")
    raw_input = payload.get("input_path")
    if not raw_input:
        raise ValueError("preview requires 'input_path'")
    input_path = Path(str(raw_input)).expanduser()
    info = probe_video(input_path)
    default_start = min(2.0, max(0.0, info.duration * 0.2)) if info.duration > 2.5 else 0.0
    default_dur = min(2.0, max(0.5, info.duration)) if info.duration > 0 else 2.0
    _run_comparison_job(
        payload,
        emitter,
        mode="preview",
        default_duration=default_dur,
        default_bitrate_kbps=3000,
        default_output_dir=_preview_cache_dir() / f"preview_{job_id[:8]}",
        default_start_sec=default_start,
        fast_baseline=True,
    )


BRIDGE_COMMANDS: dict[str, BridgeCommandHandler] = {
    "info": _handle_info,
    "probe": _handle_probe,
    "analyze": _handle_analyze,
    "enhance": _handle_enhance,
    "benchmark": _handle_benchmark,
    "preview": _handle_preview,
}


def run_bridge_command(
    command: str,
    payload: dict[str, Any],
    *,
    emitter: BridgeEmitter | None = None,
) -> int:
    """Execute a registered bridge command and emit structured NDJSON events."""
    active_emitter = emitter or BridgeEmitter()
    handler = BRIDGE_COMMANDS.get(command)
    if handler is None:
        active_emitter.emit(
            "error",
            job_id=payload.get("job_id"),
            message=f"Unknown bridge command '{command}'",
        )
        return 2

    try:
        handler(payload, active_emitter)
        return 0
    except Exception as exc:
        active_emitter.emit(
            "error",
            job_id=payload.get("job_id"),
            message=str(exc),
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: `python -m ruby_ai.bridge <command>` reading optional JSON on stdin."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        BridgeEmitter().emit("error", message="Missing bridge command")
        return 2

    command = args[0]
    raw_stdin = sys.stdin.read().strip() if sys.stdin is not None else ""
    if raw_stdin:
        try:
            payload = json.loads(raw_stdin)
            if not isinstance(payload, dict):
                raise ValueError("Bridge stdin payload must be a JSON object")
        except Exception as exc:
            BridgeEmitter().emit("error", message=f"Invalid bridge JSON payload: {exc}")
            return 2
    else:
        payload = {}

    return run_bridge_command(command, payload)


if __name__ == "__main__":
    raise SystemExit(main())
