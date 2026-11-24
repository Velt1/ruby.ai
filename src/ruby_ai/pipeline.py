"""End-to-end 5-stage video restoration and benchmarking pipelines."""

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

from ruby_ai.config import EnhanceConfig, resolve_target_dimensions
from ruby_ai.device import select_device
from ruby_ai.ffmpeg import (
    encode_baseline_lanczos,
    extract_comparison_frame,
    open_decode_pipe,
    open_encode_pipe,
    probe_video,
    resolve_codec,
    resolve_hwaccel,
    simulate_social_compression,
)
from ruby_ai.models import Restorer

ProgressCallback = Callable[[int, int | None], None]


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
    )
    return list(restored)


def _run_sync_loop(
    decode_stdout: BinaryIO,
    encode_stdin: BinaryIO,
    frame_bytes: int,
    target_w: int,
    target_h: int,
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
            encode_stdin.write(memoryview(frame))
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
            np.frombuffer(buf, dtype=np.uint8).reshape((target_h, target_w, 3))
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
    encode_stdin: BinaryIO,
    frame_bytes: int,
    target_w: int,
    target_h: int,
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
        raise RuntimeError("Pipeline aborted due to worker error")

    def decode_worker() -> None:
        try:
            while not stop_event.is_set():
                buf = _read_exact_into(decode_stdout, frame_bytes)
                if buf is None:
                    break
                item: np.ndarray | bytearray = (
                    np.frombuffer(buf, dtype=np.uint8).reshape((target_h, target_w, 3))
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
                encode_stdin.write(memoryview(item))
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
                raise RuntimeError("Encoder worker stopped unexpectedly")
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


def enhance_video(
    input_path: Path,
    output_path: Path,
    config: EnhanceConfig,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    show_progress: bool = True,
    on_progress: ProgressCallback | None = None,
) -> EnhanceResult:
    """Run temporal pre-denoise, supersampling, 1x neural restoration, and social encoding."""
    info = probe_video(input_path)
    target_w, target_h = resolve_target_dimensions(
        info.display_width,
        info.display_height,
        config.max_short_edge,
    )

    restorer: Restorer | None = None
    device_backend: str | None = None
    device_label = "ffmpeg-only (passthrough)"
    if config.ai_strength > 0.0:
        device_info = select_device(config.device)
        device_backend = device_info.backend
        restorer = Restorer(
            config.model,
            device_info,
            fp16=config.fp16,
            optimize_gpu=config.optimize_gpu,
            compile_model=config.compile_model,
            allow_unsafe_pickle=config.allow_unsafe_pickle,
        )
        if restorer.dtype == torch.float16:
            precision = "fp16"
        elif restorer.dtype == torch.bfloat16:
            precision = "bf16"
        else:
            precision = "fp32"
        device_label = f"{device_info.name} ({precision})"
    elif config.codec == "auto" or config.hwaccel == "auto":
        device_backend = select_device(config.device).backend

    resolved_codec = resolve_codec(config.codec, device_backend=device_backend)
    resolved_hwaccel = resolve_hwaccel(config.hwaccel, device_backend=device_backend)

    if duration_sec is not None:
        total_frames: int | None = max(1, round(duration_sec * info.fps_num / info.fps_den))
    elif info.frame_count > 0:
        total_frames = info.frame_count
    else:
        total_frames = None

    frame_bytes = target_w * target_h * 3
    decode_proc = open_decode_pipe(
        info,
        config,
        target_w,
        target_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
        hwaccel=resolved_hwaccel,
    )
    encode_proc = open_encode_pipe(
        info,
        output_path,
        config,
        target_w,
        target_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
        resolved_codec=resolved_codec,
    )

    started = time.perf_counter()
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
                f"Processing {target_w}x{target_h} [{device_label} | {resolved_codec}]",
                total=total_frames,
            )
            loop_fn = _run_async_loop if config.async_io else _run_sync_loop
            frames_processed = loop_fn(
                decode_proc.proc.stdout,
                encode_proc.proc.stdin,
                frame_bytes,
                target_w,
                target_h,
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
    except Exception:
        decode_proc.terminate()
        encode_proc.terminate()
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
        codec_used=resolved_codec,
        effective_tile_size=restorer.last_tile_size if restorer is not None else None,
        batch_size=config.batch_size,
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
) -> BenchmarkReport:
    """Generate baseline Lanczos, ruby.ai, and simulated social-compressed comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(input_path)
    target_w, target_h = resolve_target_dimensions(
        info.display_width,
        info.display_height,
        config.max_short_edge,
    )

    baseline_path = output_dir / "baseline_lanczos.mp4"
    enhanced_path = output_dir / "ruby_enhanced.mp4"
    baseline_social_path = output_dir / "baseline_lanczos_social.mp4"
    enhanced_social_path = output_dir / "ruby_enhanced_social.mp4"

    encode_baseline_lanczos(
        input_path,
        baseline_path,
        target_w=target_w,
        target_h=target_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
        crf=config.crf,
    )
    enhance_result = enhance_video(
        input_path,
        enhanced_path,
        config,
        start_sec=start_sec,
        duration_sec=duration_sec,
        show_progress=show_progress,
        on_progress=on_progress,
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
