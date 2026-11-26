"""Extensible GPU post-processing pipeline (Wavelet Color Fix, Detail Recovery, Dehalo)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

_KERNEL_CACHE: dict[tuple[float, int, str, torch.dtype], torch.Tensor] = {}


@runtime_checkable
class FramePostProcessor(Protocol):
    """Protocol for extensible GPU NCHW frame post-processors."""

    name: str

    def process(self, restored: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Transform `restored` NCHW tensor in [0, 1] using `reference` and return NCHW tensor."""
        ...


def _get_gaussian_kernel_1d(
    sigma: float,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (round(float(sigma), 4), int(kernel_size), str(device), dtype)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    half = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - half
    kernel = torch.exp(-0.5 * (coords / max(1e-4, sigma)) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    if len(_KERNEL_CACHE) > 64:
        _KERNEL_CACHE.clear()
    _KERNEL_CACHE[key] = kernel
    return kernel


def _gaussian_blur_2d(
    x: torch.Tensor,
    sigma: float,
    kernel_size: int | None = None,
) -> torch.Tensor:
    """Apply fast separable depthwise Gaussian blur to an NCHW float tensor."""
    if sigma <= 0.0:
        return x
    ksize = kernel_size if kernel_size is not None else max(3, int(round(sigma * 3.0)) * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    _, channels, height, width = x.shape
    pad = ksize // 2
    if pad == 0:
        return x

    kernel_1d = _get_gaussian_kernel_1d(sigma, ksize, x.device, x.dtype)
    kh = kernel_1d.view(1, 1, 1, ksize).expand(channels, 1, 1, ksize)
    kv = kernel_1d.view(1, 1, ksize, 1).expand(channels, 1, ksize, 1)

    pad_mode_h = "reflect" if pad < width else "replicate"
    pad_mode_v = "reflect" if pad < height else "replicate"
    blurred = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode=pad_mode_h), kh, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (0, 0, pad, pad), mode=pad_mode_v), kv, groups=channels)
    return blurred


def _rgb_to_ycbcr_bt709(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert NCHW float RGB [0,1] into BT.709 (Y, Cb, Cr) planes."""
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    return y, cb, cr


def _ycbcr_to_rgb_bt709(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    """Convert BT.709 (Y, Cb, Cr) planes back to NCHW float RGB."""
    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return torch.cat((r, g, b), dim=1)


class WaveletColorFixProcessor:
    """Topaz keepColor / WaveletFix equivalent: preserves reference color & global exposure."""

    name = "wavelet_color_fix"

    def __init__(
        self,
        *,
        strength: float = 1.0,
        radius: float = 10.0,
        luma_anchor: float = 0.35,
    ) -> None:
        self.strength = float(strength)
        self.radius = float(radius)
        self.luma_anchor = float(luma_anchor)

    def process(self, restored: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0.0:
            return restored

        orig_dtype = restored.dtype
        res_f = restored.float()
        ref_f = reference.float()

        # Downsample first when computing large-radius low-frequency envelopes for speed.
        _, _, h, w = res_f.shape
        ds_scale = 2 if min(h, w) >= 64 else 1
        res_y, res_cb, res_cr = _rgb_to_ycbcr_bt709(res_f)
        ref_y, ref_cb, ref_cr = _rgb_to_ycbcr_bt709(ref_f)

        ycc_res = torch.cat((res_y, res_cb, res_cr), dim=1)
        ycc_ref = torch.cat((ref_y, ref_cb, ref_cr), dim=1)

        if ds_scale > 1:
            sigma = max(1.0, self.radius / ds_scale)
            low_res = F.interpolate(
                _gaussian_blur_2d(F.avg_pool2d(ycc_res, ds_scale), sigma),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            low_ref = F.interpolate(
                _gaussian_blur_2d(F.avg_pool2d(ycc_ref, ds_scale), sigma),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            low_res = _gaussian_blur_2d(ycc_res, max(1.0, self.radius * 0.5))
            low_ref = _gaussian_blur_2d(ycc_ref, max(1.0, self.radius * 0.5))

        delta = low_ref - low_res
        y_fixed = res_y + (self.strength * self.luma_anchor) * delta[:, 0:1]
        cb_fixed = res_cb + self.strength * delta[:, 1:2]
        cr_fixed = res_cr + self.strength * delta[:, 2:3]

        return _ycbcr_to_rgb_bt709(y_fixed, cb_fixed, cr_fixed).clamp(0.0, 1.0).to(orig_dtype)


class DetailRecoveryProcessor:
    """Topaz Recover Details equivalent: restores fine high-frequency luma texture with coring."""

    name = "detail_recovery"

    def __init__(
        self,
        *,
        strength: float = 0.2,
        sigma: float = 1.2,
        coring: float = 0.008,
        shadow_floor: float = 0.08,
    ) -> None:
        self.strength = float(strength)
        self.sigma = float(sigma)
        self.coring = float(coring)
        self.shadow_floor = float(shadow_floor)

    def process(self, restored: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0.0:
            return restored

        orig_dtype = restored.dtype
        res_f = restored.float()
        ref_f = reference.float()

        res_y, res_cb, res_cr = _rgb_to_ycbcr_bt709(res_f)
        ref_y, _, _ = _rgb_to_ycbcr_bt709(ref_f)

        ref_low = _gaussian_blur_2d(ref_y, self.sigma)
        orig_hf = ref_y - ref_low

        # Soft-threshold (coring) suppresses sensor grain while preserving true texture edges.
        cored_hf = torch.sign(orig_hf) * F.relu(orig_hf.abs() - self.coring)
        # Gate down dark shadow regions so high-ISO shadow noise is not re-injected.
        shadow_gate = ((ref_y - self.shadow_floor) / 0.15).clamp(0.0, 1.0)
        y_out = (res_y + self.strength * shadow_gate * cored_hf).clamp(0.0, 1.0)

        return _ycbcr_to_rgb_bt709(y_out, res_cb, res_cr).clamp(0.0, 1.0).to(orig_dtype)


class DehaloProcessor:
    """Clamps unnatural bright/dark AI ringing halos outside local reference luma extrema."""

    name = "dehalo"

    def __init__(self, *, strength: float = 0.0, kernel_size: int = 3) -> None:
        self.strength = float(strength)
        self.kernel_size = int(kernel_size)

    def process(self, restored: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0.0:
            return restored

        orig_dtype = restored.dtype
        res_f = restored.float()
        ref_f = reference.float()

        res_y, res_cb, res_cr = _rgb_to_ycbcr_bt709(res_f)
        ref_y, _, _ = _rgb_to_ycbcr_bt709(ref_f)

        pad = self.kernel_size // 2
        local_max = F.max_pool2d(ref_y, self.kernel_size, stride=1, padding=pad)
        local_min = -F.max_pool2d(-ref_y, self.kernel_size, stride=1, padding=pad)

        # Allow natural detail contrast expansion (1.5% margin), damp only true ringing overshoot.
        margin = 0.015
        clamped_y = res_y.clamp(local_min - margin, local_max + margin)
        y_out = res_y * (1.0 - self.strength) + clamped_y * self.strength
        return _ycbcr_to_rgb_bt709(y_out, res_cb, res_cr).clamp(0.0, 1.0).to(orig_dtype)


class PostProcessPipeline:
    """Composable sequence of FramePostProcessor stages."""

    def __init__(self, processors: Sequence[FramePostProcessor] = ()) -> None:
        self.processors: list[FramePostProcessor] = list(processors)

    def __bool__(self) -> bool:
        return len(self.processors) > 0

    def process(self, restored: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        out = restored
        for proc in self.processors:
            out = proc.process(out, reference)
        return out


def build_postprocess_pipeline(
    *,
    keep_color: bool = True,
    recover_details: float = 0.0,
    dehalo: float = 0.0,
) -> PostProcessPipeline:
    """Construct a standard post-processing pipeline from config parameters."""
    stages: list[FramePostProcessor] = []
    if keep_color:
        stages.append(WaveletColorFixProcessor(strength=1.0))
    if dehalo > 0.0:
        stages.append(DehaloProcessor(strength=dehalo))
    if recover_details > 0.0:
        stages.append(DetailRecoveryProcessor(strength=recover_details))
    return PostProcessPipeline(stages)
