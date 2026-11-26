"""End-to-end streaming video restoration, upscaling, and benchmarking pipelines."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ruby_ai.config import EnhanceConfig, _round_even, resolve_target_dimensions
from ruby_ai.device import select_device
from ruby_ai.ffmpeg import (
    ManagedProcess,
    encode_baseline_lanczos,
    extract_comparison_frame,
    open_decode_pipe,
    open_encode_pipe,
    probe_video,
    resolve_codec,
    resolve_hwaccel,
    simulate_social_compression,
)
from ruby_ai.models import Restorer, get_model_spec_scale

ProgressCallback = Callable[[int, int | None], None]
StatusCallback = Callable[[str, str, str | None], None]


@dataclass(frozen=True, slots=True)
class EnhanceResult:
    """Summary metrics produced by an enhancement run."""

    input_path: Path
    output_path: Path
    frames_processed: int
    elapsed_sec: float
    fps: float
    device_label: str
    target_width: int
    target_height: int
    model_key: str = ""
    codec_used: str = ""
    effective_tile_size: int | None = None
    batch_size: int = 1
    effective_tile_batch_size: int = 1
    pipe_bit_depth: int = 8
    model_scale: int = 1
    hwaccel_used: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Artifacts and throughput metrics produced by a benchmark run."""

    artifacts: dict[str, Path]
    enhance_result: EnhanceResult
    bitrate_kbps: int


def _read_exact_into(stream: BinaryIO, size: int) -> bytearray | None:
    """Read exactly `size` bytes into a writable bytearray unless clean EOF is reached."""
    buf = bytearray(size)
    view = memoryview(buf)
    offset = 0
    while offset < size:
        chunk = stream.read(size - offset)
        if not chunk:
            if offset == 0:
                return None
            raise RuntimeError(
                f"Incomplete raw frame received from FFmpeg decode ({offset} of {size} bytes)."
            )
        n = len(chunk)
        view[offset : offset + n] = chunk
        offset += n
    return buf


def _buffer_to_frame(
    buf: bytearray,
    *,
    width: int,
    height: int,
    bit_depth: int,
) -> np.ndarray:
    dtype = np.dtype("<u2") if bit_depth == 16 else np.uint8
    return np.frombuffer(buf, dtype=dtype).reshape((height, width, 3))


def _flush_batch(
    batch_items: list[np.ndarray | bytearray],
    restorer: Restorer | None,
    config: EnhanceConfig,
) -> list[np.ndarray | bytearray]:
    if not batch_items:
        return []
    if restorer is None:
        return batch_items
    frames = [item for item in batch_items if isinstance(item, np.ndarray)]
    restored = restorer.restore_batch(
        frames,
        ai_strength=config.ai_strength,
        tile_size=config.tile_size,
        tile_overlap=config.tile_overlap,
        tile_batch_size=config.tile_batch_size,
        tile_blending=config.tile_blending,
        keep_color=config.keep_color,
        recover_details=config.recover_details,
        dehalo=config.dehalo,
        temporal_consistency=config.temporal_consistency,
        scene_change_threshold=config.scene_change_threshold,
    )
    return list(restored)


def _write_frame_safe(encode_proc: ManagedProcess, frame: np.ndarray | bytearray) -> None:
    assert encode_proc.proc.stdin is not None
    try:
        encode_proc.proc.stdin.write(memoryview(frame).cast("B"))
    except (BrokenPipeError, OSError) as exc:
        detail = encode_proc.tail_stderr()
        raise RuntimeError(
            f"FFmpeg encode pipe closed unexpectedly: {detail or exc}"
        ) from exc


def _run_sync_loop(
    decode_stdout: BinaryIO,
    encode_proc: ManagedProcess,
    frame_bytes: int,
    decode_w: int,
    decode_h: int,
    restorer: Restorer | None,
    config: EnhanceConfig,
    total_frames: int | None,
    progress: Progress,
    task_id: Any,
    on_progress: ProgressCallback | None,
) -> int:
    frames_processed = 0
    batch: list[np.ndarray | bytearray] = []

    def emit(items: list[np.ndarray | bytearray]) -> None:
        nonlocal frames_processed
        processed = _flush_batch(items, restorer, config)
        for frame in processed:
            _write_frame_safe(encode_proc, frame)
        count = len(processed)
        frames_processed += count
        progress.update(task_id, advance=count)
        if on_progress is not None:
            on_progress(frames_processed, total_frames)

    while True:
        buf = _read_exact_into(decode_stdout, frame_bytes)
        if buf is None:
            break
        item: np.ndarray | bytearray = (
            _buffer_to_frame(
                buf, width=decode_w, height=decode_h, bit_depth=config.pipe_bit_depth
            )
            if restorer is not None
            else buf
        )
        batch.append(item)
        if len(batch) >= config.batch_size:
            emit(batch)
            batch.clear()

    if batch:
        emit(batch)
        batch.clear()

    return frames_processed


def _run_async_loop(
    decode_stdout: BinaryIO,
    encode_proc: ManagedProcess,
    frame_bytes: int,
    decode_w: int,
    decode_h: int,
    restorer: Restorer | None,
    config: EnhanceConfig,
    total_frames: int | None,
    progress: Progress,
    task_id: Any,
    on_progress: ProgressCallback | None,
) -> int:
    queue_cap = max(4, config.batch_size * 2)
    decode_queue: queue.Queue[np.ndarray | bytearray | None] = queue.Queue(maxsize=queue_cap)
    encode_queue: queue.Queue[np.ndarray | bytearray | None] = queue.Queue(maxsize=queue_cap)
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []
    err_lock = threading.Lock()

    def record_error(exc: BaseException) -> None:
        with err_lock:
            worker_errors.append(exc)
        stop_event.set()

    def put_item(q: queue.Queue[Any], item: Any) -> bool:
        while not stop_event.is_set():
            try:
                q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def get_item(q: queue.Queue[Any]) -> Any:
        while not stop_event.is_set():
            try:
                return q.get(timeout=0.05)
            except queue.Empty:
                continue
        with err_lock:
            if worker_errors:
                raise worker_errors[0]
        raise RuntimeError("Pipeline aborted due to worker error")

    def decode_worker() -> None:
        try:
            while not stop_event.is_set():
                buf = _read_exact_into(decode_stdout, frame_bytes)
                if buf is None:
                    break
                item: np.ndarray | bytearray = (
                    _buffer_to_frame(
                        buf, width=decode_w, height=decode_h, bit_depth=config.pipe_bit_depth
                    )
                    if restorer is not None
                    else buf
                )
                if not put_item(decode_queue, item):
                    break
        except BaseException as exc:
            record_error(exc)
        finally:
            if not stop_event.is_set():
                put_item(decode_queue, None)

    def encode_worker() -> None:
        try:
            while True:
                item = get_item(encode_queue)
                if item is None:
                    break
                _write_frame_safe(encode_proc, item)
        except BaseException as exc:
            if not stop_event.is_set():
                record_error(exc)

    dec_thread = threading.Thread(target=decode_worker, daemon=True)
    enc_thread = threading.Thread(target=encode_worker, daemon=True)
    dec_thread.start()
    enc_thread.start()

    frames_processed = 0
    batch: list[np.ndarray | bytearray] = []

    def emit(items: list[np.ndarray | bytearray]) -> None:
        nonlocal frames_processed
        processed = _flush_batch(items, restorer, config)
        for frame in processed:
            if not put_item(encode_queue, frame):
                with err_lock:
                    if worker_errors:
                        raise worker_errors[0]
                detail = encode_proc.tail_stderr()
                raise RuntimeError(
                    f"Encoder worker stopped unexpectedly: {detail}"
                    if detail
                    else "Encoder worker stopped unexpectedly"
                )
        count = len(processed)
        frames_processed += count
        progress.update(task_id, advance=count)
        if on_progress is not None:
            on_progress(frames_processed, total_frames)

    try:
        while True:
            item = get_item(decode_queue)
            if item is None:
                break
            batch.append(item)
            if len(batch) >= config.batch_size:
                emit(batch)
                batch.clear()

        if batch:
            emit(batch)
            batch.clear()

        put_item(encode_queue, None)
        dec_thread.join()
        enc_thread.join()
    except BaseException as exc:
        stop_event.set()
        dec_thread.join(timeout=2.0)
        enc_thread.join(timeout=2.0)
        if worker_errors:
            raise worker_errors[0] from exc
        raise

    if worker_errors:
        raise worker_errors[0]

    return frames_processed


def _software_fallback_for(codec: str) -> str | None:
    if codec == "h264_nvenc" or codec == "h264_videotoolbox":
        return "libx264"
    if codec == "hevc_nvenc":
        return "libx265"
    return None


def enhance_video(
    input_path: Path,
    output_path: Path,
    config: EnhanceConfig,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    show_progress: bool = True,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
) -> EnhanceResult:
    """Run temporal pre-denoise, supersampling/upscaling, neural restoration, and encoding."""
    if on_status is not None:
        on_status("probe", "Prüfe Video-Metadaten…", input_path.name)
    info = probe_video(input_path)

    model_scale = get_model_spec_scale(config.model)
    restorer: Restorer | None = None
    device_backend: str | None = None
    device_label = "ffmpeg-only (passthrough)"
    need_neural = config.ai_strength > 0.0 or model_scale > 1

    if need_neural:
        device_info = select_device(config.device)
        device_backend = device_info.backend
        restorer = Restorer(
            config.model,
            device_info,
            fp16=config.fp16,
            optimize_gpu=config.optimize_gpu,
            compile_model=config.compile_model,
            allow_unsafe_pickle=config.allow_unsafe_pickle,
            on_status=on_status,
        )
        model_scale = restorer.scale
        if restorer.dtype == torch.float16:
            precision = "fp16"
        elif restorer.dtype == torch.bfloat16:
            precision = "bf16"
        else:
            precision = "fp32"
        device_label = f"{device_info.name} ({precision})"
    elif config.codec == "auto" or config.hwaccel == "auto":
        device_backend = select_device(config.device).backend

    allow_upscale = model_scale > 1
    target_w, target_h = resolve_target_dimensions(
        info.display_width,
        info.display_height,
        config.max_short_edge,
        allow_upscale=allow_upscale,
        model_scale=model_scale,
    )

    if model_scale > 1:
        decode_w = min(info.display_width, _round_even(target_w / model_scale))
        decode_h = min(info.display_height, _round_even(target_h / model_scale))
    else:
        decode_w, decode_h = target_w, target_h

    net_out_w = decode_w * model_scale
    net_out_h = decode_h * model_scale

    resolved_codec = resolve_codec(config.codec, device_backend=device_backend)
    resolved_hwaccel = resolve_hwaccel(config.hwaccel, device_backend=device_backend)

    if duration_sec is not None:
        total_frames: int | None = max(1, round(duration_sec * info.fps_num / info.fps_den))
    elif info.frame_count > 0:
        total_frames = info.frame_count
    else:
        total_frames = None

    bytes_per_channel = 2 if config.pipe_bit_depth == 16 else 1
    frame_bytes = decode_w * decode_h * 3 * bytes_per_channel

    def _execute_pipes(active_codec: str, active_hwaccel: str | None) -> int:
        if restorer is not None:
            restorer.reset_temporal()

        decode_proc = open_decode_pipe(
            info,
            config,
            decode_w,
            decode_h,
            start_sec=start_sec,
            duration_sec=duration_sec,
            hwaccel=active_hwaccel,
        )
        encode_proc = open_encode_pipe(
            info,
            output_path,
            config,
            target_w,
            target_h,
            input_w=net_out_w,
            input_h=net_out_h,
            start_sec=start_sec,
            duration_sec=duration_sec,
            resolved_codec=active_codec,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not show_progress,
        )

        try:
            assert decode_proc.proc.stdout is not None
            assert encode_proc.proc.stdin is not None
            with progress:
                task_id = progress.add_task(
                    f"Processing {target_w}x{target_h} [{device_label} | {active_codec}]",
                    total=total_frames,
                )
                loop_fn = _run_async_loop if config.async_io else _run_sync_loop
                count = loop_fn(
                    decode_proc.proc.stdout,
                    encode_proc,
                    frame_bytes,
                    decode_w,
                    decode_h,
                    restorer,
                    config,
                    total_frames,
                    progress,
                    task_id,
                    on_progress,
                )

            encode_proc.close_stdin()
            decode_proc.wait_and_check("FFmpeg decode")
            encode_proc.wait_and_check("FFmpeg encode")
            return count
        except Exception:
            decode_proc.terminate()
            encode_proc.terminate()
            raise

    if on_status is not None:
        on_status(
            "enhance",
            (
                f"Verarbeite Video ({target_w}×{target_h}, "
                f"{config.pipe_bit_depth}-Bit, {resolved_codec})…"
            ),
            device_label,
        )

    started = time.perf_counter()
    active_codec = resolved_codec
    active_hwaccel = resolved_hwaccel

    for attempt in range(3):
        try:
            frames_processed = _execute_pipes(active_codec, active_hwaccel)
            break
        except Exception as exc:
            msg = str(exc)
            # Retry on CPU decode if auto CUDA decode fails on 10-bit/4:2:2/ProRes sources.
            if (
                attempt < 2
                and active_hwaccel == "cuda"
                and config.hwaccel == "auto"
                and "FFmpeg decode" in msg
            ):
                active_hwaccel = None
                if on_status is not None:
                    on_status(
                        "fallback",
                        (
                            "CUDA-Hardware-Decode nicht unterstützt – "
                            "wechsle automatisch zu CPU-Decode…"
                        ),
                        None,
                    )
                continue

            sw_fallback = _software_fallback_for(active_codec)
            if (
                attempt < 2
                and sw_fallback is not None
                and config.codec == "auto"
                and ("FFmpeg encode" in msg or "encode pipe" in msg.lower())
            ):
                active_codec = sw_fallback
                if on_status is not None:
                    on_status(
                        "fallback",
                        f"Hardware-Encoder fehlgeschlagen – wechsle zu '{active_codec}'…",
                        None,
                    )
                continue
            raise

    if frames_processed == 0:
        raise RuntimeError(f"No frames were decoded from '{input_path}'")

    elapsed = max(1e-6, time.perf_counter() - started)
    return EnhanceResult(
        input_path=input_path,
        output_path=output_path,
        frames_processed=frames_processed,
        elapsed_sec=elapsed,
        fps=frames_processed / elapsed,
        device_label=device_label,
        target_width=target_w,
        target_height=target_h,
        model_key=config.model,
        codec_used=active_codec,
        effective_tile_size=restorer.last_tile_size if restorer is not None else None,
        batch_size=config.batch_size,
        effective_tile_batch_size=(
            restorer.last_tile_batch_size if restorer is not None else 1
        ),
        pipe_bit_depth=config.pipe_bit_depth,
        model_scale=model_scale,
        hwaccel_used=active_hwaccel,
    )


def run_benchmark_detailed(
    input_path: Path,
    output_dir: Path,
    config: EnhanceConfig,
    *,
    start_sec: float | None = None,
    duration_sec: float = 5.0,
    bitrate_kbps: int = 3500,
    reference_path: Path | None = None,
    show_progress: bool = True,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
    fast_baseline: bool = False,
) -> BenchmarkReport:
    """Generate baseline Lanczos, ruby.ai, and simulated social-compressed comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(input_path)
    model_scale = get_model_spec_scale(config.model) if config.ai_strength > 0.0 else 1
    target_w, target_h = resolve_target_dimensions(
        info.display_width,
        info.display_height,
        config.max_short_edge,
        allow_upscale=model_scale > 1,
        model_scale=model_scale,
    )

    baseline_path = output_dir / "baseline_lanczos.mp4"
    enhanced_path = output_dir / "ruby_enhanced.mp4"
    baseline_social_path = output_dir / "baseline_lanczos_social.mp4"
    enhanced_social_path = output_dir / "ruby_enhanced_social.mp4"

    if on_status is not None:
        on_status("baseline", "Erstelle Lanczos-Vergleichsclip…", f"{target_w}×{target_h}")
    encode_baseline_lanczos(
        input_path,
        baseline_path,
        target_w=target_w,
        target_h=target_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
        crf=config.crf,
        fast_preview=fast_baseline,
        hdr_to_sdr=config.hdr_to_sdr,
        is_hdr=info.is_hdr,
        color_transfer=info.color_transfer,
    )

    enhance_result = enhance_video(
        input_path,
        enhanced_path,
        config,
        start_sec=start_sec,
        duration_sec=duration_sec,
        show_progress=show_progress,
        on_progress=on_progress,
        on_status=on_status,
    )

    if on_status is not None:
        on_status(
            "social_sim",
            f"Simuliere Instagram/TikTok-Kompression ({bitrate_kbps} kbps)…",
            None,
        )
    simulate_social_compression(baseline_path, baseline_social_path, bitrate_kbps=bitrate_kbps)
    simulate_social_compression(enhanced_path, enhanced_social_path, bitrate_kbps=bitrate_kbps)

    outputs: dict[str, Path] = {
        "baseline": baseline_path,
        "enhanced": enhanced_path,
        "baseline_social": baseline_social_path,
        "enhanced_social": enhanced_social_path,
    }
    comparison_inputs = [baseline_social_path, enhanced_social_path]

    if reference_path is not None:
        reference_social_path = output_dir / "reference_social.mp4"
        simulate_social_compression(
            reference_path,
            reference_social_path,
            bitrate_kbps=bitrate_kbps,
        )
        outputs["reference_social"] = reference_social_path
        comparison_inputs.append(reference_social_path)

    comparison_png = output_dir / "comparison_after_compression.png"
    extract_comparison_frame(
        comparison_inputs,
        comparison_png,
        timestamp_sec=max(0.1, duration_sec * 0.5),
    )
    outputs["comparison_png"] = comparison_png
    return BenchmarkReport(
        artifacts=outputs,
        enhance_result=enhance_result,
        bitrate_kbps=bitrate_kbps,
    )


def run_benchmark(
    input_path: Path,
    output_dir: Path,
    config: EnhanceConfig,
    *,
    start_sec: float | None = None,
    duration_sec: float = 5.0,
    bitrate_kbps: int = 3500,
    reference_path: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Path]:
    """Generate baseline Lanczos, ruby.ai, and simulated social-compressed comparisons."""
    return run_benchmark_detailed(
        input_path,
        output_dir,
        config,
        start_sec=start_sec,
        duration_sec=duration_sec,
        bitrate_kbps=bitrate_kbps,
        reference_path=reference_path,
        show_progress=show_progress,
    ).artifacts
