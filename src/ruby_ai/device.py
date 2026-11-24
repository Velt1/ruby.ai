"""Compute device selection and precision policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved execution device metadata."""

    device: torch.device
    backend: str
    name: str
    default_fp16: bool
    total_memory_bytes: int | None = None


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


def resolve_auto_tile_size(
    device_info: DeviceInfo,
    width: int,
    height: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Choose full-frame (0) or a safe tile size based on available VRAM and frame batch."""
    if device_info.backend == "cuda" and device_info.total_memory_bytes is not None:
        vram_gib = device_info.total_memory_bytes / (1024**3)
        pixels = max(1, width) * max(1, height) * max(1, batch_size)
        dtype_factor = 1.0 if dtype in (torch.float16, torch.bfloat16) else 2.0
        effective_pixels = pixels * dtype_factor
        if vram_gib >= 10.0 and effective_pixels <= 4_500_000:
            return 0
        if vram_gib >= 6.0 and effective_pixels <= 2_600_000:
            return 0
        if vram_gib >= 6.0:
            return 768
        return 512

    return 512


def resolve_dtype(device_info: DeviceInfo, fp16: bool | None) -> torch.dtype:
    """Determine inference dtype from explicit override or device default."""
    use_fp16 = device_info.default_fp16 if fp16 is None else fp16
    if use_fp16 and device_info.backend == "cpu":
        raise ValueError("fp16 inference is not supported on CPU")
    return torch.float16 if use_fp16 else torch.float32
