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
from ruby_ai.config import PRESETS, EnhanceConfig, PresetName
from ruby_ai.device import select_device
from ruby_ai.ffmpeg import available_h264_encoders, require_binary
from ruby_ai.models import MODEL_REGISTRY
from ruby_ai.pipeline import enhance_video, run_benchmark

app = typer.Typer(
    name="ruby-ai",
    help="Open-source 4K-to-1080p video restoration and supersampling CLI for social media.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def enhance(
    input_path: Annotated[
        Path,
        typer.Argument(help="Input video file (e.g. 4K MP4/MOV)."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output MP4 path (defaults next to input)."),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", help=f"Base preset ({', '.join(sorted(PRESETS))})."),
    ] = "crisp",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Registry model key or path to .safetensors file."),
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
            help="Blend weight between supersampled frame and 1x model (0.0-1.0).",
        ),
    ] = None,
    sharpen: Annotated[
        float | None,
        typer.Option("--sharpen", help="Adaptive CAS sharpening strength (0.0-1.0)."),
    ] = None,
    grain: Annotated[
        int | None,
        typer.Option("--grain", help="Subtle temporal luma grain (0-10) to reduce banding."),
    ] = None,
    tile_size: Annotated[
        int | None,
        typer.Option("--tile-size", help="Tile width/height in pixels (0 disables tiling)."),
    ] = None,
    tile_overlap: Annotated[
        int | None,
        typer.Option("--tile-overlap", help="Tile overlap in pixels for seamless blending."),
    ] = None,
    crf: Annotated[
        int | None,
        typer.Option("--crf", help="Target encoder CRF/CQ quality (0-51)."),
    ] = None,
    codec: Annotated[
        str | None,
        typer.Option("--codec", help="H.264 encoder: libx264, h264_nvenc, or h264_videotoolbox."),
    ] = None,
    x264_preset: Annotated[
        str | None,
        typer.Option("--x264-preset", help="libx264 speed preset (e.g. medium, slow)."),
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
    """Enhance and downscale a video for Instagram/TikTok upload."""
    if preset not in PRESETS:
        raise typer.BadParameter(
            f"Unknown preset '{preset}'. Choose from: {', '.join(sorted(PRESETS))}"
        )

    config = EnhanceConfig.from_preset(
        cast(PresetName, preset),
        model=model,
        device=device,
        fp16=fp16,
        max_short_edge=max_short_edge,
        temporal_denoise=temporal_denoise,
        ai_strength=ai_strength,
        sharpen=sharpen,
        grain=grain,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        crf=crf,
        codec=codec,
        x264_preset=x264_preset,
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    out_path = output or input_path.with_name(f"{input_path.stem}_ruby_{preset}.mp4")

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
        f"{result.frames_processed} frames in {result.elapsed_sec:.2f}s @ {result.fps:.2f} fps)"
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
    ] = "crisp",
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

    config = EnhanceConfig.from_preset(
        cast(PresetName, preset),
        device=device,
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
    target_dir = output_dir or (Path("outputs") / f"{input_path.stem}_benchmark")
    artifacts = run_benchmark(
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
    table.add_column("Path")
    for key, path in artifacts.items():
        table.add_row(key, str(path))
    console.print(table)


@app.command()
def models() -> None:
    """List built-in verified 1x restoration models and their licenses."""
    table = Table(title="Built-in 1x Restoration Models")
    table.add_column("Key", style="cyan")
    table.add_column("Architecture")
    table.add_column("Scale")
    table.add_column("Author")
    table.add_column("License", style="green")
    table.add_column("Description")
    for spec in MODEL_REGISTRY.values():
        table.add_row(
            spec.key,
            spec.architecture,
            f"{spec.scale}x",
            spec.author,
            spec.license,
            spec.description,
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Inspect Python, PyTorch GPU backend, Spandrel, and FFmpeg availability."""
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
        table.add_row("Selected Compute", f"{dev.backend} — {dev.name} ({prec})")
    except Exception as exc:
        table.add_row("Selected Compute", f"[red]{exc}[/red]")

    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    mps_backend = getattr(torch.backends, "mps", None)
    table.add_row("Apple MPS Available", str(bool(mps_backend and mps_backend.is_available())))

    for binary in ("ffmpeg", "ffprobe"):
        try:
            table.add_row(binary, require_binary(binary))
        except RuntimeError as exc:
            table.add_row(binary, f"[red]{exc}[/red]")

    try:
        encoders = available_h264_encoders()
        table.add_row("H.264 Encoders", ", ".join(encoders) if encoders else "none detected")
    except Exception as exc:
        table.add_row("H.264 Encoders", f"[red]{exc}[/red]")

    console.print(table)
