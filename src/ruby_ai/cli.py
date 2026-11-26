"""Command-line interface for ruby.ai."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated, cast

import spandrel
import torch
import typer
from rich.console import Console
from rich.table import Table

from ruby_ai import __version__
from ruby_ai.config import EXPORT_TARGETS, PRESETS, EnhanceConfig, PresetName
from ruby_ai.device import (
    detect_nvidia_gpu_hardware,
    query_cuda_memory_info,
    select_device,
)
from ruby_ai.ffmpeg import (
    available_video_encoders,
    probe_encoder_usable,
    require_binary,
)
from ruby_ai.models import MODEL_REGISTRY
from ruby_ai.pipeline import enhance_video, run_benchmark_detailed

app = typer.Typer(
    name="ruby-ai",
    help="Open-source video restoration, upscaling, and social-4K export CLI.",
    no_args_is_help=True,
)
console = Console()


def _format_tile_label(tile_size: int | None) -> str:
    if tile_size is None:
        return "n/a"
    if tile_size == 0:
        return "full-frame"
    return f"{tile_size}px"


@app.command()
def enhance(
    input_path: Annotated[
        Path,
        typer.Argument(help="Input video file (e.g. 4K MP4/MOV)."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output video path (defaults next to input)."),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", help=f"Base preset ({', '.join(sorted(PRESETS))})."),
    ] = "social-crisp",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Registry model key or path to .safetensors file."),
    ] = None,
    export_target: Annotated[
        str | None,
        typer.Option(
            "--export-target",
            help=f"Social export profile ({', '.join(sorted(EXPORT_TARGETS))}).",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Compute device: auto, cuda, mps, or cpu."),
    ] = None,
    fp16: Annotated[
        bool | None,
        typer.Option("--fp16/--no-fp16", help="Override half-precision inference."),
    ] = None,
    max_short_edge: Annotated[
        int | None,
        typer.Option("--max-short-edge", help="Target short edge in pixels (default 1080)."),
    ] = None,
    temporal_denoise: Annotated[
        float | None,
        typer.Option("--temporal-denoise", help="Pre-denoise strength via atadenoise (0.0-0.25)."),
    ] = None,
    ai_strength: Annotated[
        float | None,
        typer.Option(
            "--ai-strength",
            help="Blend weight between reference frame and neural output (0.0-1.0).",
        ),
    ] = None,
    keep_color: Annotated[
        bool | None,
        typer.Option(
            "--keep-color/--no-keep-color",
            help="Match low-frequency BT.709 color & exposure to input (WaveletFix).",
        ),
    ] = None,
    recover_details: Annotated[
        float | None,
        typer.Option(
            "--recover-details",
            help="Blend cored high-frequency original detail back into luma (0.0-1.0).",
        ),
    ] = None,
    dehalo: Annotated[
        float | None,
        typer.Option(
            "--dehalo",
            help="Suppress neural edge ringing overshoot (0.0-1.0).",
        ),
    ] = None,
    temporal_consistency: Annotated[
        float | None,
        typer.Option(
            "--temporal-consistency",
            help="Motion-compensated temporal stabilization strength (0.0-1.0).",
        ),
    ] = None,
    scene_change_threshold: Annotated[
        float | None,
        typer.Option(
            "--scene-change-threshold",
            help="Scene-cut threshold resetting temporal history (0.05-0.80).",
        ),
    ] = None,
    sharpen: Annotated[
        float | None,
        typer.Option("--sharpen", help="Luma-only adaptive CAS sharpening strength (0.0-1.0)."),
    ] = None,
    clarity: Annotated[
        float | None,
        typer.Option("--clarity", help="Luma micro-contrast / OLED clarity boost (0.0-1.0)."),
    ] = None,
    deband: Annotated[
        bool | None,
        typer.Option(
            "--deband/--no-deband",
            help="Smooth dark-gradient banding before sharpening.",
        ),
    ] = None,
    grain: Annotated[
        int | None,
        typer.Option("--grain", help="Subtle temporal luma grain (0-10) to reduce banding."),
    ] = None,
    hdr_to_sdr: Annotated[
        bool | None,
        typer.Option("--hdr-to-sdr/--no-hdr-to-sdr", help="Auto-tonemap HDR sources to BT.709."),
    ] = None,
    pipe_bit_depth: Annotated[
        int | None,
        typer.Option("--pipe-bit-depth", help="Frame pipe bit depth: 8 (rgb24) or 16 (rgb48le)."),
    ] = None,
    maxrate_kbps: Annotated[
        int | None,
        typer.Option("--maxrate-kbps", help="Optional VBV peak bitrate cap in kbps."),
    ] = None,
    gop_seconds: Annotated[
        float | None,
        typer.Option("--gop-seconds", help="Keyframe interval in seconds (0 disables fixed GOP)."),
    ] = None,
    tile_size: Annotated[
        int | None,
        typer.Option(
            "--tile-size",
            help="Tile size in pixels (auto if omitted, 0 disables tiling).",
        ),
    ] = None,
    tile_overlap: Annotated[
        int | None,
        typer.Option("--tile-overlap", help="Tile overlap in pixels for seamless blending."),
    ] = None,
    tile_batch_size: Annotated[
        int | None,
        typer.Option("--tile-batch-size", help="Number of tiles evaluated together per GPU pass."),
    ] = None,
    tile_blending: Annotated[
        str | None,
        typer.Option("--tile-blending", help="Tile overlap strategy: gaussian, cosine, or linear."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Number of video frames processed per GPU batch."),
    ] = None,
    crf: Annotated[
        int | None,
        typer.Option("--crf", help="Target encoder CRF/CQ quality (0-51)."),
    ] = None,
    codec: Annotated[
        str | None,
        typer.Option(
            "--codec",
            help="Encoder: auto, libx264, h264_nvenc, hevc_nvenc, libx265, prores_ks.",
        ),
    ] = None,
    x264_preset: Annotated[
        str | None,
        typer.Option("--x264-preset", help="libx264/libx265 speed preset (e.g. medium, slow)."),
    ] = None,
    nvenc_preset: Annotated[
        str | None,
        typer.Option("--nvenc-preset", help="NVENC quality preset (p1-p7, default p6)."),
    ] = None,
    hwaccel: Annotated[
        str | None,
        typer.Option("--hwaccel", help="FFmpeg decode acceleration: none, auto, or cuda."),
    ] = None,
    optimize_gpu: Annotated[
        bool | None,
        typer.Option(
            "--optimize-gpu/--no-optimize-gpu",
            help="Enable CUDA kernel/layout optimizations (FastGroupNorm, NHWC, TF32).",
        ),
    ] = None,
    compile_model: Annotated[
        bool | None,
        typer.Option(
            "--compile-model/--no-compile-model",
            help="Attempt torch.compile acceleration when available.",
        ),
    ] = None,
    async_io: Annotated[
        bool | None,
        typer.Option(
            "--async-io/--no-async-io",
            help="Enable async decode/GPU/encode producer-consumer queues.",
        ),
    ] = None,
    start: Annotated[
        float | None,
        typer.Option("--start", help="Optional start offset in seconds."),
    ] = None,
    duration: Annotated[
        float | None,
        typer.Option("--duration", help="Optional clip duration limit in seconds."),
    ] = None,
    allow_unsafe_pickle: Annotated[
        bool,
        typer.Option(
            "--allow-unsafe-pickle",
            help="Allow loading untrusted local .pth/.pt/.ckpt pickle files.",
        ),
    ] = False,
) -> None:
    """Enhance, restore, or upscale a video for social or master delivery."""
    if preset not in PRESETS:
        raise typer.BadParameter(
            f"Unknown preset '{preset}'. Choose from: {', '.join(sorted(PRESETS))}"
        )

    effective_edge = max_short_edge
    if effective_edge is None and export_target in EXPORT_TARGETS:
        effective_edge = EXPORT_TARGETS[export_target].get("max_short_edge")

    config = EnhanceConfig.from_preset(
        cast(PresetName, preset),
        model=model,
        export_target=export_target,
        device=device,
        fp16=fp16,
        max_short_edge=effective_edge,
        temporal_denoise=temporal_denoise,
        ai_strength=ai_strength,
        keep_color=keep_color,
        recover_details=recover_details,
        dehalo=dehalo,
        temporal_consistency=temporal_consistency,
        scene_change_threshold=scene_change_threshold,
        sharpen=sharpen,
        clarity=clarity,
        deband=deband,
        grain=grain,
        hdr_to_sdr=hdr_to_sdr,
        pipe_bit_depth=pipe_bit_depth,
        maxrate_kbps=maxrate_kbps,
        gop_seconds=gop_seconds,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        tile_blending=tile_blending,
        batch_size=batch_size,
        crf=crf,
        codec=codec,
        x264_preset=x264_preset,
        nvenc_preset=nvenc_preset,
        hwaccel=hwaccel,
        optimize_gpu=optimize_gpu,
        compile_model=compile_model,
        async_io=async_io,
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    ext = ".mov" if config.codec == "prores_ks" else ".mp4"
    out_path = output or input_path.with_name(f"{input_path.stem}_ruby_{preset}{ext}")

    result = enhance_video(
        input_path,
        out_path,
        config,
        start_sec=start,
        duration_sec=duration,
    )
    console.print(
        f"[green]Done:[/green] {result.output_path} "
        f"({result.target_width}x{result.target_height}, "
        f"{result.frames_processed} frames in {result.elapsed_sec:.2f}s @ {result.fps:.2f} fps | "
        f"codec={result.codec_used}, {result.pipe_bit_depth}-bit, "
        f"tile={_format_tile_label(result.effective_tile_size)}, "
        f"tile_batch={result.effective_tile_batch_size}, batch={result.batch_size})"
    )


@app.command()
def benchmark(
    input_path: Annotated[
        Path,
        typer.Argument(help="Input video file to benchmark."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Directory for comparison clips and PNG."),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", help="Preset to benchmark."),
    ] = "social-crisp",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override registry model key or .safetensors path."),
    ] = None,
    export_target: Annotated[
        str | None,
        typer.Option("--export-target", help="Social export profile override."),
    ] = None,
    codec: Annotated[
        str | None,
        typer.Option("--codec", help="Encoder override (auto, libx264, h264_nvenc, etc.)."),
    ] = None,
    nvenc_preset: Annotated[
        str | None,
        typer.Option("--nvenc-preset", help="NVENC quality preset override (p1-p7)."),
    ] = None,
    hwaccel: Annotated[
        str | None,
        typer.Option("--hwaccel", help="FFmpeg decode acceleration override (none, auto, cuda)."),
    ] = None,
    tile_size: Annotated[
        int | None,
        typer.Option(
            "--tile-size",
            help="Tile size override (auto if omitted, 0 disables tiling).",
        ),
    ] = None,
    tile_overlap: Annotated[
        int | None,
        typer.Option("--tile-overlap", help="Tile overlap override in pixels."),
    ] = None,
    tile_batch_size: Annotated[
        int | None,
        typer.Option("--tile-batch-size", help="Tiles evaluated together per GPU pass."),
    ] = None,
    tile_blending: Annotated[
        str | None,
        typer.Option("--tile-blending", help="Tile blending strategy: gaussian, cosine, linear."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Frames processed per GPU batch."),
    ] = None,
    temporal_denoise: Annotated[
        float | None,
        typer.Option("--temporal-denoise", help="Pre-denoise strength override (0.0-0.25)."),
    ] = None,
    ai_strength: Annotated[
        float | None,
        typer.Option("--ai-strength", help="AI blend weight override (0.0-1.0)."),
    ] = None,
    keep_color: Annotated[
        bool | None,
        typer.Option("--keep-color/--no-keep-color", help="Toggle WaveletFix color lock."),
    ] = None,
    recover_details: Annotated[
        float | None,
        typer.Option("--recover-details", help="Detail recovery strength (0.0-1.0)."),
    ] = None,
    dehalo: Annotated[
        float | None,
        typer.Option("--dehalo", help="Dehalo suppression strength (0.0-1.0)."),
    ] = None,
    temporal_consistency: Annotated[
        float | None,
        typer.Option("--temporal-consistency", help="Temporal stabilization strength (0.0-1.0)."),
    ] = None,
    scene_change_threshold: Annotated[
        float | None,
        typer.Option("--scene-change-threshold", help="Scene-cut gate threshold (0.05-0.80)."),
    ] = None,
    sharpen: Annotated[
        float | None,
        typer.Option("--sharpen", help="Luma CAS sharpening override (0.0-1.0)."),
    ] = None,
    clarity: Annotated[
        float | None,
        typer.Option("--clarity", help="Luma micro-contrast / clarity override (0.0-1.0)."),
    ] = None,
    deband: Annotated[
        bool | None,
        typer.Option("--deband/--no-deband", help="Toggle dark-gradient debanding."),
    ] = None,
    grain: Annotated[
        int | None,
        typer.Option("--grain", help="Temporal luma grain override (0-10)."),
    ] = None,
    hdr_to_sdr: Annotated[
        bool | None,
        typer.Option("--hdr-to-sdr/--no-hdr-to-sdr", help="Toggle HDR->SDR tonemapping."),
    ] = None,
    pipe_bit_depth: Annotated[
        int | None,
        typer.Option("--pipe-bit-depth", help="Frame pipe bit depth: 8 or 16."),
    ] = None,
    maxrate_kbps: Annotated[
        int | None,
        typer.Option("--maxrate-kbps", help="Optional VBV peak bitrate cap in kbps."),
    ] = None,
    gop_seconds: Annotated[
        float | None,
        typer.Option("--gop-seconds", help="Keyframe interval in seconds."),
    ] = None,
    crf: Annotated[
        int | None,
        typer.Option("--crf", help="Target encoder CRF/CQ override (0-51)."),
    ] = None,
    optimize_gpu: Annotated[
        bool | None,
        typer.Option("--optimize-gpu/--no-optimize-gpu", help="Toggle CUDA GPU optimizations."),
    ] = None,
    async_io: Annotated[
        bool | None,
        typer.Option("--async-io/--no-async-io", help="Toggle async decode/encode queues."),
    ] = None,
    start: Annotated[
        float | None,
        typer.Option("--start", help="Optional start offset in seconds."),
    ] = None,
    duration: Annotated[
        float,
        typer.Option("--duration", help="Clip segment length in seconds."),
    ] = 5.0,
    bitrate_kbps: Annotated[
        int,
        typer.Option("--bitrate-kbps", help="Simulated social-platform H.264 bitrate in kbps."),
    ] = 3500,
    reference: Annotated[
        Path | None,
        typer.Option("--reference", help="Optional third-party reference clip for comparison."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Compute device override."),
    ] = None,
    allow_unsafe_pickle: Annotated[
        bool,
        typer.Option(
            "--allow-unsafe-pickle",
            help="Allow loading untrusted local .pth/.pt/.ckpt pickle files.",
        ),
    ] = False,
) -> None:
    """Compare Lanczos baseline vs. ruby.ai (and optional reference) after social re-encoding."""
    if preset not in PRESETS:
        raise typer.BadParameter(
            f"Unknown preset '{preset}'. Choose from: {', '.join(sorted(PRESETS))}"
        )

    effective_edge = (
        EXPORT_TARGETS[export_target].get("max_short_edge")
        if export_target in EXPORT_TARGETS
        else None
    )

    config = EnhanceConfig.from_preset(
        cast(PresetName, preset),
        model=model,
        export_target=export_target,
        max_short_edge=effective_edge,
        codec=codec,
        nvenc_preset=nvenc_preset,
        hwaccel=hwaccel,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        tile_blending=tile_blending,
        batch_size=batch_size,
        temporal_denoise=temporal_denoise,
        ai_strength=ai_strength,
        keep_color=keep_color,
        recover_details=recover_details,
        dehalo=dehalo,
        temporal_consistency=temporal_consistency,
        scene_change_threshold=scene_change_threshold,
        sharpen=sharpen,
        clarity=clarity,
        deband=deband,
        grain=grain,
        hdr_to_sdr=hdr_to_sdr,
        pipe_bit_depth=pipe_bit_depth,
        maxrate_kbps=maxrate_kbps,
        gop_seconds=gop_seconds,
        crf=crf,
        optimize_gpu=optimize_gpu,
        async_io=async_io,
        device=device,
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    target_dir = output_dir or (Path("outputs") / f"{input_path.stem}_benchmark")
    report = run_benchmark_detailed(
        input_path,
        target_dir,
        config,
        start_sec=start,
        duration_sec=duration,
        bitrate_kbps=bitrate_kbps,
        reference_path=reference,
    )
    table = Table(title=f"Benchmark Outputs ({bitrate_kbps} kbps simulation)")
    table.add_column("Artifact", style="cyan")
    table.add_column("Size (MiB)", justify="right")
    table.add_column("Path")
    for key, path in report.artifacts.items():
        size_mib = path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0
        table.add_row(key, f"{size_mib:.2f}", str(path))
    console.print(table)

    res = report.enhance_result
    console.print(
        f"[green]Enhanced Performance:[/green] model={res.model_key}, codec={res.codec_used}, "
        f"{res.target_width}x{res.target_height}, {res.frames_processed} frames in "
        f"{res.elapsed_sec:.2f}s @ [bold]{res.fps:.2f} fps[/bold] "
        f"[{res.device_label}, tile={_format_tile_label(res.effective_tile_size)}, "
        f"tile_batch={res.effective_tile_batch_size}, batch={res.batch_size}]"
    )


@app.command()
def models() -> None:
    """List built-in verified 1x & 2x restoration/upscaling models and their licenses."""
    table = Table(title="Built-in Restoration & Upscaling Models")
    table.add_column("Key", style="cyan")
    table.add_column("Architecture")
    table.add_column("Scale")
    table.add_column("Source")
    table.add_column("Author")
    table.add_column("License")
    table.add_column("Description")
    for spec in MODEL_REGISTRY.values():
        lic_style = "green" if spec.commercial_use else "yellow"
        source_label = "HF" if spec.source.source_kind == "hf" else "URL"
        table.add_row(
            spec.key,
            spec.architecture,
            f"{spec.scale}x",
            source_label,
            spec.author,
            f"[{lic_style}]{spec.license}[/{lic_style}]",
            spec.description,
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Inspect Python, PyTorch GPU backend, free VRAM, Spandrel, and FFmpeg encoders."""
    table = Table(title=f"ruby.ai v{__version__} Environment Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    table.add_row("Platform", f"{platform.system()} {platform.machine()}")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Spandrel", getattr(spandrel, "__version__", "installed"))

    try:
        dev = select_device("auto")
        prec = "fp16 default" if dev.default_fp16 else "fp32 default"
        mem_info = query_cuda_memory_info(dev)
        vram_suffix = (
            f", {dev.total_memory_bytes / (1024**3):.2f} GiB VRAM"
            if dev.total_memory_bytes is not None
            else ""
        )
        if mem_info is not None:
            vram_suffix += f" ({mem_info[0] / (1024**3):.2f} GiB free)"
        table.add_row("Selected Compute", f"{dev.backend} — {dev.name} ({prec}{vram_suffix})")
    except Exception as exc:
        table.add_row("Selected Compute", f"[red]{exc}[/red]")

    cuda_avail = bool(torch.cuda.is_available())
    table.add_row("CUDA Available", str(cuda_avail))
    nv_hw = detect_nvidia_gpu_hardware()
    if nv_hw is not None:
        hw_label = f"{nv_hw['name']} (Driver {nv_hw['driver_version']})"
        if not cuda_avail:
            hw_label = f"[yellow]{hw_label} — CUDA inactive, update NVIDIA driver![/yellow]"
        table.add_row("NVIDIA Hardware", hw_label)

    mps_backend = getattr(torch.backends, "mps", None)
    table.add_row("Apple MPS Available", str(bool(mps_backend and mps_backend.is_available())))

    for binary in ("ffmpeg", "ffprobe"):
        try:
            table.add_row(binary, require_binary(binary))
        except RuntimeError as exc:
            table.add_row(binary, f"[red]{exc}[/red]")

    try:
        encoders = available_video_encoders()
        table.add_row("Video Encoders", ", ".join(encoders) if encoders else "none detected")
        if "h264_nvenc" in encoders:
            nvenc_ok, nvenc_err = probe_encoder_usable("h264_nvenc")
            nvenc_status = (
                "[green]usable[/green]"
                if nvenc_ok
                else f"[yellow]failed runtime probe ({nvenc_err})[/yellow]"
            )
            table.add_row("NVENC Runtime", nvenc_status)
    except Exception as exc:
        table.add_row("Video Encoders", f"[red]{exc}[/red]")

    console.print(table)
