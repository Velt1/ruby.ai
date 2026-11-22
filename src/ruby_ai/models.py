"""Open-weight 1x restoration model registry, safe loading, and tiled inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spandrel
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ruby_ai.device import DeviceInfo, resolve_dtype

PICKLE_SUFFIXES = frozenset({".pth", ".pt", ".ckpt", ".bin"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata and attribution for a verified open-weight restoration model."""

    key: str
    repo_id: str
    filename: str
    architecture: str
    scale: int
    author: str
    license: str
    description: str
    source_url: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "de-h264": ModelSpec(
        key="de-h264",
        repo_id="Phips/1xDeH264_realplksr",
        filename="1xDeH264_realplksr.safetensors",
        architecture="RealPLKSR",
        scale=1,
        author="Philip Hofmann (Phips)",
        license="CC-BY-4.0",
        description=(
            "1x restoration removing H.264 compression artifacts and preserving fine detail."
        ),
        source_url="https://huggingface.co/Phips/1xDeH264_realplksr",
    ),
    "denoise": ModelSpec(
        key="denoise",
        repo_id="Phips/1xDeNoise_realplksr_otf",
        filename="1xDeNoise_realplksr_otf.safetensors",
        architecture="RealPLKSR",
        scale=1,
        author="Philip Hofmann (Phips)",
        license="CC-BY-4.0",
        description="1x denoising for noisy, high-ISO, or low-light live-action footage.",
        source_url="https://huggingface.co/Phips/1xDeNoise_realplksr_otf",
    ),
    "de-jpg": ModelSpec(
        key="de-jpg",
        repo_id="Phips/1xDeJPG_realplksr_otf",
        filename="1xDeJPG_realplksr_otf.safetensors",
        architecture="RealPLKSR",
        scale=1,
        author="Philip Hofmann (Phips)",
        license="CC-BY-4.0",
        description="1x de-artifacting for blocky or heavily recompressed video sources.",
        source_url="https://huggingface.co/Phips/1xDeJPG_realplksr_otf",
    ),
}


def resolve_model_path(
    model_ref: str,
    *,
    allow_unsafe_pickle: bool = False,
) -> tuple[Path, ModelSpec | None]:
    """Resolve a registry key (downloading if needed) or validate a local weight file."""
    if model_ref in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[model_ref]
        downloaded = hf_hub_download(repo_id=spec.repo_id, filename=spec.filename)
        return Path(downloaded), spec

    path = Path(model_ref).expanduser()
    if not path.is_file():
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise FileNotFoundError(
            f"Model '{model_ref}' is neither a known registry key ({known}) "
            "nor an existing file."
        )

    suffix = path.suffix.lower()
    if suffix in PICKLE_SUFFIXES and not allow_unsafe_pickle:
        raise ValueError(
            f"Refusing to load pickle checkpoint '{path.name}' without --allow-unsafe-pickle. "
            "Prefer .safetensors weights."
        )
    if suffix != ".safetensors" and suffix not in PICKLE_SUFFIXES:
        raise ValueError(
            f"Unsupported model file extension '{suffix}'. Expected .safetensors."
        )

    return path, None


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


def _pad_for_model(
    tensor: torch.Tensor,
    *,
    multiple_of: int,
    minimum: int,
) -> tuple[torch.Tensor, int, int]:
    """Pad NCHW tensor on bottom/right edges to satisfy model size constraints."""
    _, _, height, width = tensor.shape
    mult = max(1, multiple_of)
    min_dim = max(0, minimum)

    target_h = max(min_dim, ((height + mult - 1) // mult) * mult)
    target_w = max(min_dim, ((width + mult - 1) // mult) * mult)
    pad_h = target_h - height
    pad_w = target_w - width

    if pad_h == 0 and pad_w == 0:
        return tensor, height, width

    mode = "reflect" if (pad_h < height and pad_w < width) else "replicate"
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, height, width


def _feather_mask_1d(
    length: int,
    *,
    fade_left: int,
    fade_right: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a 1D blending weight vector with smooth linear ramps on interior borders."""
    mask = torch.ones(length, dtype=torch.float32, device=device)
    left = min(fade_left, length // 2)
    right = min(fade_right, length // 2)
    if left > 0:
        mask[:left] = (torch.arange(left, dtype=torch.float32, device=device) + 0.5) / left
    if right > 0:
        mask[-right:] = (
            torch.arange(right - 1, -1, -1, dtype=torch.float32, device=device) + 0.5
        ) / right
    return mask


def infer_tiled(
    fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    *,
    tile_size: int,
    tile_overlap: int,
    multiple_of: int = 1,
    minimum: int = 0,
) -> torch.Tensor:
    """Run a 1x NCHW model function over an image, using feathered tiles when needed."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected NCHW tensor with batch=1, got shape {tuple(image.shape)}")

    _, channels, height, width = image.shape
    if tile_size == 0 or (height <= tile_size and width <= tile_size):
        padded, orig_h, orig_w = _pad_for_model(
            image, multiple_of=multiple_of, minimum=minimum
        )
        out = fn(padded)
        return out[:, :, :orig_h, :orig_w]

    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("tile_overlap must be >= 0 and strictly smaller than tile_size")

    ys = _tile_starts(height, tile_size, tile_overlap)
    xs = _tile_starts(width, tile_size, tile_overlap)

    output_acc = torch.zeros((1, channels, height, width), dtype=torch.float32, device=image.device)
    weight_acc = torch.zeros((1, 1, height, width), dtype=torch.float32, device=image.device)

    for y0 in ys:
        y1 = min(y0 + tile_size, height)
        fade_top = tile_overlap if y0 > 0 else 0
        fade_bottom = tile_overlap if y1 < height else 0
        wy = _feather_mask_1d(
            y1 - y0, fade_left=fade_top, fade_right=fade_bottom, device=image.device
        )

        for x0 in xs:
            x1 = min(x0 + tile_size, width)
            fade_left = tile_overlap if x0 > 0 else 0
            fade_right = tile_overlap if x1 < width else 0
            wx = _feather_mask_1d(
                x1 - x0, fade_left=fade_left, fade_right=fade_right, device=image.device
            )

            patch = image[:, :, y0:y1, x0:x1]
            padded_patch, patch_h, patch_w = _pad_for_model(
                patch, multiple_of=multiple_of, minimum=minimum
            )
            restored_patch = fn(padded_patch)[:, :, :patch_h, :patch_w]

            weight_2d = (wy[:, None] * wx[None, :]).view(1, 1, patch_h, patch_w)
            output_acc[:, :, y0:y1, x0:x1] += restored_patch.to(torch.float32) * weight_2d
            weight_acc[:, :, y0:y1, x0:x1] += weight_2d

    return (output_acc / weight_acc.clamp_min(1e-6)).to(image.dtype)


class Restorer:
    """Loads a 1x Spandrel restoration model and restores uint8 RGB video frames."""

    def __init__(
        self,
        model_ref: str,
        device_info: DeviceInfo,
        *,
        fp16: bool | None = None,
        allow_unsafe_pickle: bool = False,
    ) -> None:
        self.model_path, self.spec = resolve_model_path(
            model_ref, allow_unsafe_pickle=allow_unsafe_pickle
        )
        self.device_info = device_info
        self.dtype = resolve_dtype(device_info, fp16)

        descriptor = spandrel.ModelLoader(device=torch.device("cpu")).load_from_file(
            self.model_path
        )
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise TypeError(
                f"Expected an ImageModelDescriptor, got {type(descriptor).__name__}"
            )
        if descriptor.scale != 1:
            raise ValueError(
                f"Expected a 1x restoration model, got scale={descriptor.scale}x. "
                "Downscale first and use a 1x model."
            )

        self.descriptor = descriptor
        self.model = descriptor.model.to(device=device_info.device, dtype=self.dtype)
        self.model.eval()

        size_req = getattr(descriptor, "size_requirements", None)
        self.multiple_of = int(getattr(size_req, "multiple_of", 1) or 1)
        self.minimum = int(getattr(size_req, "minimum", 0) or 0)

    def restore_frame(
        self,
        frame_rgb_u8: np.ndarray,
        *,
        ai_strength: float,
        tile_size: int,
        tile_overlap: int,
    ) -> np.ndarray:
        """Restore a single HWC uint8 RGB frame and blend with the input by ai_strength."""
        if ai_strength <= 0.0:
            return frame_rgb_u8

        writable = np.require(frame_rgb_u8, requirements=["C", "W"])
        tensor = (
            torch.from_numpy(writable)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device_info.device, dtype=self.dtype)
            .div(255.0)
        )

        with torch.inference_mode():
            restored = infer_tiled(
                self.model,
                tensor,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                multiple_of=self.multiple_of,
                minimum=self.minimum,
            )
            blended = (
                tensor * (1.0 - ai_strength) + restored * ai_strength
                if ai_strength < 1.0
                else restored
            )
            out_u8 = (
                blended.clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .squeeze(0)
                .permute(1, 2, 0)
                .to(dtype=torch.uint8, device="cpu")
                .numpy()
            )

        return np.ascontiguousarray(out_u8)
