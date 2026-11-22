"""End-to-end 5-stage video restoration and benchmarking pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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
    simulate_social_compression,
)
from ruby_ai.models import Restorer


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


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly `size` bytes unless EOF is reached."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def enhance_video(
    input_path: Path,
    output_path: Path,
    config: EnhanceConfig,
    *,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    show_progress: bool = True,
) -> EnhanceResult:
    """Run temporal pre-denoise, supersampling, 1x neural restoration, and social encoding."""
    info = probe_video(input_path)
    target_w, target_h = resolve_target_dimensions(
        info.display_width,
        info.display_height,
        config.max_short_edge,
    )

    restorer: Restorer | None = None
    device_label = "ffmpeg-only (passthrough)"
    if config.ai_strength > 0.0:
        device_info = select_device(config.device)
        restorer = Restorer(
            config.model,
            device_info,
            fp16=config.fp16,
            allow_unsafe_pickle=config.allow_unsafe_pickle,
        )
        precision = "fp16" if restorer.dtype == torch.float16 else "fp32"
        device_label = f"{device_info.name} ({precision})"

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
    )
    encode_proc = open_encode_pipe(
        info,
        output_path,
        config,
        target_w,
        target_h,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )

    frames_processed = 0
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
                f"Processing {target_w}x{target_h} [{device_label}]",
                total=total_frames,
            )
            while True:
                raw = _read_exact(decode_proc.proc.stdout, frame_bytes)
                if not raw:
                    break
                if len(raw) < frame_bytes:
                    raise RuntimeError(
                        f"Incomplete raw frame received from FFmpeg decode "
                        f"({len(raw)} of {frame_bytes} bytes)."
                    )

                if restorer is not None:
                    frame_u8 = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w, 3))
                    restored_u8 = restorer.restore_frame(
                        frame_u8,
                        ai_strength=config.ai_strength,
                        tile_size=config.tile_size,
                        tile_overlap=config.tile_overlap,
                    )
                    encode_proc.proc.stdin.write(restored_u8.tobytes())
                else:
                    encode_proc.proc.stdin.write(raw)

                frames_processed += 1
                progress.update(task_id, advance=1)

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
    enhance_video(
        input_path,
        enhanced_path,
        config,
        start_sec=start_sec,
        duration_sec=duration_sec,
        show_progress=show_progress,
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
    return outputs
