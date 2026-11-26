"""Generic tile splitting, multi-scale execution, and 2D blending strategies."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class TileBlendingStrategy(Protocol):
    """Protocol for computing normalized 2D blending weights for overlapping tiles."""

    name: str

    def make_weight_2d(
        self,
        tile_h: int,
        tile_w: int,
        *,
        fade_top: int,
        fade_bottom: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a (tile_h, tile_w) float32 weight tensor on `device`."""
        ...


def _clamp_margins(length: int, fade_left: int, fade_right: int) -> tuple[int, int]:
    left = min(max(0, fade_left), length // 2)
    right = min(max(0, fade_right), length // 2)
    return left, right


class GaussianBlendingStrategy:
    """Separable 2D Gaussian feathering on tile overlap borders (Topaz-style)."""

    name = "gaussian"

    def __init__(self, var_factor: float = 1.0) -> None:
        if var_factor <= 0.0:
            raise ValueError("var_factor must be strictly positive")
        self.var_factor = float(var_factor)

    def _ramp_1d(
        self,
        length: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(length, dtype=torch.float32, device=device)
        left, right = _clamp_margins(length, fade_left, fade_right)
        scale = 2.5 / max(0.05, self.var_factor)
        if left > 0:
            idx = torch.arange(left, dtype=torch.float32, device=device)
            u = (left - 0.5 - idx) / left
            mask[:left] = torch.exp(-0.5 * (u * scale) ** 2)
        if right > 0:
            idx = torch.arange(right, dtype=torch.float32, device=device)
            u = (idx + 0.5) / right
            mask[-right:] = torch.exp(-0.5 * (u * scale) ** 2)
        return mask

    def make_weight_2d(
        self,
        tile_h: int,
        tile_w: int,
        *,
        fade_top: int,
        fade_bottom: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        wy = self._ramp_1d(tile_h, fade_top, fade_bottom, device)
        wx = self._ramp_1d(tile_w, fade_left, fade_right, device)
        return wy[:, None] * wx[None, :]


class CosineBlendingStrategy:
    """Separable raised-cosine blending for C1-smooth overlap transitions."""

    name = "cosine"

    def _ramp_1d(
        self,
        length: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(length, dtype=torch.float32, device=device)
        left, right = _clamp_margins(length, fade_left, fade_right)
        if left > 0:
            t = (torch.arange(left, dtype=torch.float32, device=device) + 0.5) / left
            mask[:left] = 0.5 * (1.0 - torch.cos(math.pi * t))
        if right > 0:
            t = (torch.arange(right, dtype=torch.float32, device=device) + 0.5) / right
            mask[-right:] = 0.5 * (1.0 + torch.cos(math.pi * t))
        return mask

    def make_weight_2d(
        self,
        tile_h: int,
        tile_w: int,
        *,
        fade_top: int,
        fade_bottom: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        wy = self._ramp_1d(tile_h, fade_top, fade_bottom, device)
        wx = self._ramp_1d(tile_w, fade_left, fade_right, device)
        return wy[:, None] * wx[None, :]


class LinearBlendingStrategy:
    """Separable linear ramp blending across tile overlap borders."""

    name = "linear"

    def _ramp_1d(
        self,
        length: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(length, dtype=torch.float32, device=device)
        left, right = _clamp_margins(length, fade_left, fade_right)
        if left > 0:
            mask[:left] = (torch.arange(left, dtype=torch.float32, device=device) + 0.5) / left
        if right > 0:
            mask[-right:] = (
                torch.arange(right - 1, -1, -1, dtype=torch.float32, device=device) + 0.5
            ) / right
        return mask

    def make_weight_2d(
        self,
        tile_h: int,
        tile_w: int,
        *,
        fade_top: int,
        fade_bottom: int,
        fade_left: int,
        fade_right: int,
        device: torch.device,
    ) -> torch.Tensor:
        wy = self._ramp_1d(tile_h, fade_top, fade_bottom, device)
        wx = self._ramp_1d(tile_w, fade_left, fade_right, device)
        return wy[:, None] * wx[None, :]


class TileBlendingRegistry:
    """Extensible registry of tile blending strategies."""

    def __init__(self, strategies: Sequence[TileBlendingStrategy] = ()) -> None:
        self._strategies: dict[str, TileBlendingStrategy] = {}
        for strategy in strategies:
            self.register(strategy)

    def register(self, strategy: TileBlendingStrategy, *, overwrite: bool = False) -> None:
        key = strategy.name.strip().lower()
        if not key:
            raise ValueError("Blending strategy name must be non-empty")
        if not overwrite and key in self._strategies:
            raise ValueError(f"Blending strategy '{key}' is already registered")
        self._strategies[key] = strategy

    def get(self, name: str) -> TileBlendingStrategy | None:
        return self._strategies.get(name.strip().lower())

    def keys(self) -> list[str]:
        return list(self._strategies.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self._strategies)


TILE_BLENDING_STRATEGIES = TileBlendingRegistry(
    [
        GaussianBlendingStrategy(),
        CosineBlendingStrategy(),
        LinearBlendingStrategy(),
    ]
)


def resolve_blending_strategy(
    blending: str | TileBlendingStrategy,
    *,
    registry: TileBlendingRegistry = TILE_BLENDING_STRATEGIES,
) -> TileBlendingStrategy:
    """Resolve a blending strategy name or validate a custom TileBlendingStrategy instance."""
    if isinstance(blending, str):
        resolved = registry.get(blending)
        if resolved is None:
            known = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Unknown tile blending strategy '{blending}' (allowed: {known})")
        return resolved
    return blending


def _tile_starts(total: int, tile_size: int, overlap: int) -> list[int]:
    """Compute sorted tile start coordinates covering [0, total) cleanly."""
    if total <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be strictly smaller than tile_size")

    starts: list[int] = []
    pos = 0
    while pos + tile_size < total:
        starts.append(pos)
        pos += stride
    last = total - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _padded_shape(height: int, width: int, *, multiple_of: int, minimum: int) -> tuple[int, int]:
    mult = max(1, multiple_of)
    min_dim = max(0, minimum)
    target_h = max(min_dim, ((height + mult - 1) // mult) * mult)
    target_w = max(min_dim, ((width + mult - 1) // mult) * mult)
    return target_h, target_w


def _pad_for_model(
    tensor: torch.Tensor,
    *,
    multiple_of: int,
    minimum: int,
) -> tuple[torch.Tensor, int, int]:
    """Pad NCHW tensor on bottom/right edges to satisfy model size constraints."""
    _, _, height, width = tensor.shape
    target_h, target_w = _padded_shape(height, width, multiple_of=multiple_of, minimum=minimum)
    pad_h = target_h - height
    pad_w = target_w - width

    if pad_h == 0 and pad_w == 0:
        return tensor, height, width

    mode = "reflect" if (pad_h < height and pad_w < width) else "replicate"
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, height, width


def infer_tiled(
    fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    *,
    tile_size: int,
    tile_overlap: int,
    multiple_of: int = 1,
    minimum: int = 0,
    tile_batch_size: int = 4,
    scale: int = 1,
    blending: str | TileBlendingStrategy = "gaussian",
) -> torch.Tensor:
    """Run an NCHW model function (1x/2x/4x) over a batch of images with feathered tiles."""
    if image.ndim != 4 or image.shape[0] < 1:
        raise ValueError(f"Expected NCHW tensor with batch>=1, got shape {tuple(image.shape)}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    batch, channels, height, width = image.shape
    if tile_size == 0 or (height <= tile_size and width <= tile_size):
        padded, orig_h, orig_w = _pad_for_model(
            image, multiple_of=multiple_of, minimum=minimum
        )
        out = fn(padded)
        return out[:, :, : orig_h * scale, : orig_w * scale]

    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("tile_overlap must be >= 0 and strictly smaller than tile_size")

    strategy = resolve_blending_strategy(blending)
    is_cl = image.is_contiguous(memory_format=torch.channels_last)
    ys = _tile_starts(height, tile_size, tile_overlap)
    xs = _tile_starts(width, tile_size, tile_overlap)

    out_h = height * scale
    out_w = width * scale
    output_acc = torch.zeros(
        (batch, channels, out_h, out_w), dtype=torch.float32, device=image.device
    )
    weight_acc = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=image.device)

    grouped_tiles: dict[
        tuple[int, int], list[tuple[int, int, int, int, int, int, int, int]]
    ] = {}
    for y0 in ys:
        y1 = min(y0 + tile_size, height)
        fade_top = tile_overlap if y0 > 0 else 0
        fade_bottom = tile_overlap if y1 < height else 0
        for x0 in xs:
            x1 = min(x0 + tile_size, width)
            fade_left = tile_overlap if x0 > 0 else 0
            fade_right = tile_overlap if x1 < width else 0
            pad_key = _padded_shape(y1 - y0, x1 - x0, multiple_of=multiple_of, minimum=minimum)
            grouped_tiles.setdefault(pad_key, []).append(
                (y0, y1, x0, x1, fade_top, fade_bottom, fade_left, fade_right)
            )

    weight_cache: dict[tuple[int, int, int, int, int, int], torch.Tensor] = {}
    effective_tile_batch = max(1, tile_batch_size // batch)
    for tiles in grouped_tiles.values():
        for start in range(0, len(tiles), effective_tile_batch):
            chunk = tiles[start : start + effective_tile_batch]
            padded_patches: list[torch.Tensor] = []
            patch_dims: list[tuple[int, int]] = []
            for y0, y1, x0, x1, _, _, _, _ in chunk:
                patch = image[:, :, y0:y1, x0:x1]
                padded_patch, patch_h, patch_w = _pad_for_model(
                    patch, multiple_of=multiple_of, minimum=minimum
                )
                padded_patches.append(padded_patch)
                patch_dims.append((patch_h, patch_w))

            batch_in = padded_patches[0] if len(padded_patches) == 1 else torch.cat(padded_patches)
            if is_cl:
                batch_in = batch_in.contiguous(memory_format=torch.channels_last)

            batch_out = fn(batch_in)
            outputs = (batch_out,) if len(chunk) == 1 else batch_out.split(batch, dim=0)

            for tile_meta, (patch_h, patch_w), restored_patch in zip(
                chunk, patch_dims, outputs, strict=True
            ):
                y0, y1, x0, x1, f_top, f_bot, f_left, f_right = tile_meta
                out_ph = patch_h * scale
                out_pw = patch_w * scale
                oy0, oy1 = y0 * scale, y1 * scale
                ox0, ox1 = x0 * scale, x1 * scale

                cropped = restored_patch[:, :, :out_ph, :out_pw]
                w_key = (
                    out_ph,
                    out_pw,
                    f_top * scale,
                    f_bot * scale,
                    f_left * scale,
                    f_right * scale,
                )
                weight_2d = weight_cache.get(w_key)
                if weight_2d is None:
                    w_raw = strategy.make_weight_2d(
                        out_ph,
                        out_pw,
                        fade_top=f_top * scale,
                        fade_bottom=f_bot * scale,
                        fade_left=f_left * scale,
                        fade_right=f_right * scale,
                        device=image.device,
                    )
                    weight_2d = w_raw.view(1, 1, out_ph, out_pw)
                    weight_cache[w_key] = weight_2d

                output_acc[:, :, oy0:oy1, ox0:ox1] += cropped.to(torch.float32) * weight_2d
                weight_acc[:, :, oy0:oy1, ox0:ox1] += weight_2d

    return (output_acc / weight_acc.clamp_min(1e-6)).to(image.dtype)
