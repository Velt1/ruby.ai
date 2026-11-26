"""Compute device selection, live VRAM inspection, and adaptive tiling policy."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved execution device metadata."""

    device: torch.device
    backend: str
    name: str
    default_fp16: bool
    total_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class TilingPlan:
    """Resolved tile size and safe concurrent tile batching plan."""

    tile_size: int
    tile_batch_size: int
    reason: str


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def select_device(requested: str = "auto") -> DeviceInfo:
    """Select the best available PyTorch compute backend or validate an explicit choice."""
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device '{requested}' (expected auto, cuda, mps, cpu)")

    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            device=torch.device("cuda", idx),
            backend="cuda",
            name=torch.cuda.get_device_name(idx),
            default_fp16=True,
            total_memory_bytes=int(props.total_memory),
        )

    if requested == "mps" or (requested == "auto" and _mps_is_available()):
        if not _mps_is_available():
            raise RuntimeError("MPS device requested, but Apple MPS is not available")
        return DeviceInfo(
            device=torch.device("mps"),
            backend="mps",
            name="Apple Silicon MPS",
            # Keep float32 by default on MPS to avoid half-precision kernel discrepancies.
            default_fp16=False,
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        name="CPU",
        default_fp16=False,
    )


def query_cuda_memory_info(device_info: DeviceInfo) -> tuple[int, int] | None:
    """Return live (free_bytes, total_bytes) on CUDA without raising on unsupported drivers."""
    if device_info.backend != "cuda" or not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info(device_info.device)
        return int(free_b), int(total_b)
    except Exception:
        return None


def detect_nvidia_gpu_hardware() -> dict[str, Any] | None:
    """Inspect NVIDIA hardware and driver version via nvidia-smi even if PyTorch CUDA fails."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": 3,
        "check": False,
    }
    if platform.system() == "Windows":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x0800_0000)
    try:
        proc = subprocess.run(
            [
                nvsmi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            **kwargs,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first_line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 2:
        return None
    vram_mb: int | None = None
    if len(parts) >= 3:
        try:
            vram_mb = int(float(parts[2]))
        except ValueError:
            vram_mb = None
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "vram_mb": vram_mb,
    }


def resolve_auto_tiling_plan(
    device_info: DeviceInfo,
    width: int,
    height: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
    model_scale: int = 1,
    requested_tile_size: int | None = None,
    requested_tile_batch_size: int = 4,
    free_memory_bytes: int | None = None,
) -> TilingPlan:
    """Compute a WDDM-safe tile size and tile batch size based on live VRAM and scale."""
    req_tb = max(1, int(requested_tile_batch_size))
    if device_info.backend != "cuda" or device_info.total_memory_bytes is None:
        tile = requested_tile_size if requested_tile_size is not None else 512
        return TilingPlan(tile_size=tile, tile_batch_size=1, reason="non-cuda-safe")

    total_gib = device_info.total_memory_bytes / (1024**3)
    free_gib = (free_memory_bytes / (1024**3)) if free_memory_bytes is not None else None
    usable_gib = min(total_gib, free_gib) if free_gib is not None else total_gib

    scale_factor = max(1.0, float(model_scale * model_scale) * 0.75) if model_scale > 1 else 1.0
    dtype_factor = 1.0 if dtype in (torch.float16, torch.bfloat16) else 2.0
    pixels = max(1, width) * max(1, height) * max(1, batch_size)
    effective_pixels = pixels * dtype_factor * scale_factor

    if requested_tile_size is not None:
        tile_size = requested_tile_size
        reason = "user-override"
    elif total_gib >= 11.5 and (free_gib is None or free_gib >= 7.5):
        if effective_pixels <= 4_500_000 and (
            free_gib is None or free_gib >= 8.5 or effective_pixels <= 2_600_000
        ):
            tile_size = 0
            reason = "full-frame-12gb+"
        else:
            tile_size = 512 if model_scale > 1 else 768
            reason = "tiled-12gb+"
    elif total_gib >= 7.0:
        # 8 GB GPUs (e.g. RTX 3080 Laptop) lose 1-2 GB to Windows WDDM; avoid full-frame spills.
        if (
            model_scale == 1
            and effective_pixels <= 1_100_000
            and (free_gib is None or free_gib >= 5.0)
        ):
            tile_size = 0
            reason = "full-frame-8gb-small"
        elif free_gib is not None and free_gib < 3.5:
            tile_size = 384
            reason = "tiled-8gb-low-free-vram"
        else:
            tile_size = 512
            reason = "tiled-8gb-wddm-safe"
    else:
        if (free_gib is not None and free_gib < 3.0) or model_scale > 1:
            tile_size = 384
            reason = "tiled-6gb-or-upscale"
        else:
            tile_size = 512
            reason = "tiled-6gb"

    if tile_size == 0:
        return TilingPlan(tile_size=0, tile_batch_size=1, reason=reason)

    if usable_gib >= 11.0:
        pixel_budget = 1_600_000.0
    elif usable_gib >= 7.0 and (free_gib is None or free_gib >= 4.5):
        pixel_budget = 700_000.0
    else:
        pixel_budget = 350_000.0

    per_tile_px = float(tile_size * tile_size) * max(1, batch_size) * dtype_factor * scale_factor
    max_safe_tb = max(1, int(pixel_budget // max(1.0, per_tile_px)))
    if free_gib is not None and free_gib < 4.5:
        max_safe_tb = 1

    return TilingPlan(
        tile_size=tile_size,
        tile_batch_size=min(req_tb, max_safe_tb),
        reason=reason,
    )


def resolve_auto_tile_size(
    device_info: DeviceInfo,
    width: int,
    height: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
    model_scale: int = 1,
    free_memory_bytes: int | None = None,
) -> int:
    """Choose full-frame (0) or a safe tile size based on available VRAM and frame batch."""
    return resolve_auto_tiling_plan(
        device_info,
        width,
        height,
        batch_size=batch_size,
        dtype=dtype,
        model_scale=model_scale,
        free_memory_bytes=free_memory_bytes,
    ).tile_size


def resolve_dtype(device_info: DeviceInfo, fp16: bool | None) -> torch.dtype:
    """Determine inference dtype from explicit override or device default."""
    use_fp16 = device_info.default_fp16 if fp16 is None else fp16
    if use_fp16 and device_info.backend == "cpu":
        raise ValueError("fp16 inference is not supported on CPU")
    return torch.float16 if use_fp16 else torch.float32
