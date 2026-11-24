"""Open-weight 1x restoration model registry, safe loading, and batched GPU inference."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import spandrel
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ruby_ai.device import DeviceInfo, resolve_auto_tile_size, resolve_dtype

PICKLE_SUFFIXES = frozenset({".pth", ".pt", ".ckpt", ".bin"})

ALLOWED_MODEL_LICENSES = frozenset(
    {
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC-BY-NC-SA-4.0",
        "MIT",
        "Apache-2.0",
        "CC0-1.0",
    }
)


def get_model_cache_dir() -> Path:
    """Return local cache directory for direct-downloaded model weights."""
    env_dir = os.environ.get("RUBY_AI_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "ruby-ai" / "models"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@runtime_checkable
class ModelSource(Protocol):
    """Protocol for extensible model weight sources."""

    filename: str
    source_kind: str

    def resolve(self) -> Path:
        """Download (if needed) and return a local path to the model weights."""
        ...


@dataclass(frozen=True, slots=True)
class HuggingFaceSource:
    """Model weight source resolved through Hugging Face Hub."""

    repo_id: str
    filename: str
    source_kind: str = "hf"

    def resolve(self) -> Path:
        return Path(hf_hub_download(repo_id=self.repo_id, filename=self.filename))


@dataclass(frozen=True, slots=True)
class DirectUrlSource:
    """Model weight source downloaded via HTTPS and cached locally."""

    url: str
    filename: str
    sha256: str | None = None
    source_kind: str = "url"

    def resolve(self) -> Path:
        cache_dir = get_model_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / self.filename
        expected = self.sha256.lower() if self.sha256 else None

        if dest.is_file() and dest.stat().st_size > 0:
            if expected is None or _sha256_file(dest) == expected:
                return dest
            dest.unlink(missing_ok=True)

        part = cache_dir / f"{self.filename}.{os.getpid()}.part"
        req = urllib.request.Request(self.url, headers={"User-Agent": "ruby-ai/0.1"})
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
            actual = digest.hexdigest()
            if expected is not None and actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for '{self.filename}': expected {expected}, got {actual}"
                )
            os.replace(part, dest)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        return dest


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata and attribution for a verified open-weight restoration model."""

    key: str
    source: ModelSource
    architecture: str
    scale: int
    author: str
    license: str
    description: str
    source_url: str
    commercial_use: bool = True

    @property
    def filename(self) -> str:
        return self.source.filename

    @property
    def repo_id(self) -> str | None:
        return self.source.repo_id if isinstance(self.source, HuggingFaceSource) else None


class ModelRegistry:
    """Extensible registry mapping model keys to ModelSpec descriptors."""

    def __init__(self, specs: Sequence[ModelSpec] = ()) -> None:
        self._models: dict[str, ModelSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModelSpec, *, overwrite: bool = False) -> ModelSpec:
        if not overwrite and spec.key in self._models:
            raise ValueError(f"Model key '{spec.key}' is already registered")
        self._models[spec.key] = spec
        return spec

    def get(self, key: str) -> ModelSpec | None:
        return self._models.get(key)

    def keys(self) -> list[str]:
        return list(self._models.keys())

    def values(self) -> list[ModelSpec]:
        return list(self._models.values())

    def items(self) -> list[tuple[str, ModelSpec]]:
        return list(self._models.items())

    def __contains__(self, key: object) -> bool:
        return key in self._models

    def __getitem__(self, key: str) -> ModelSpec:
        return self._models[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)


MODEL_REGISTRY = ModelRegistry(
    [
        ModelSpec(
            key="de-h264",
            source=HuggingFaceSource(
                repo_id="Phips/1xDeH264_realplksr",
                filename="1xDeH264_realplksr.safetensors",
            ),
            architecture="RealPLKSR",
            scale=1,
            author="Philip Hofmann (Phips)",
            license="CC-BY-4.0",
            description=(
                "1x restoration removing H.264 compression artifacts and preserving fine detail."
            ),
            source_url="https://huggingface.co/Phips/1xDeH264_realplksr",
            commercial_use=True,
        ),
        ModelSpec(
            key="denoise",
            source=HuggingFaceSource(
                repo_id="Phips/1xDeNoise_realplksr_otf",
                filename="1xDeNoise_realplksr_otf.safetensors",
            ),
            architecture="RealPLKSR",
            scale=1,
            author="Philip Hofmann (Phips)",
            license="CC-BY-4.0",
            description="1x denoising for noisy, high-ISO, or low-light live-action footage.",
            source_url="https://huggingface.co/Phips/1xDeNoise_realplksr_otf",
            commercial_use=True,
        ),
        ModelSpec(
            key="de-jpg",
            source=HuggingFaceSource(
                repo_id="Phips/1xDeJPG_realplksr_otf",
                filename="1xDeJPG_realplksr_otf.safetensors",
            ),
            architecture="RealPLKSR",
            scale=1,
            author="Philip Hofmann (Phips)",
            license="CC-BY-4.0",
            description="1x de-artifacting for blocky or heavily recompressed video sources.",
            source_url="https://huggingface.co/Phips/1xDeJPG_realplksr_otf",
            commercial_use=True,
        ),
        ModelSpec(
            key="superscale-span",
            source=DirectUrlSource(
                url="https://github.com/Kim2091/Kim2091-Models/releases/download/1x-SuperScale/1x-SuperScale_SPAN.safetensors",
                filename="1x-SuperScale_SPAN.safetensors",
            ),
            architecture="SPAN",
            scale=1,
            author="Kim2091",
            license="CC-BY-NC-SA-4.0",
            description=(
                "Fast 1x SPAN detail/anti-aliasing restoration (non-commercial license)."
            ),
            source_url="https://openmodeldb.info/models/1x-SuperScale",
            commercial_use=False,
        ),
        ModelSpec(
            key="superscale-fast",
            source=DirectUrlSource(
                url="https://github.com/Kim2091/Kim2091-Models/releases/download/1x-SuperScale/1x-SuperScale_RPLKSR_S.safetensors",
                filename="1x-SuperScale_RPLKSR_S.safetensors",
            ),
            architecture="RealPLKSR-S",
            scale=1,
            author="Kim2091",
            license="CC-BY-NC-SA-4.0",
            description=(
                "Compact RealPLKSR-S 1x photo/video restoration (non-commercial license)."
            ),
            source_url="https://openmodeldb.info/models/1x-SuperScale-RPLKSR-S",
            commercial_use=False,
        ),
        ModelSpec(
            key="de-jpg-fast",
            source=HuggingFaceSource(
                repo_id="Phips/1xDeJPG_OmniSR",
                filename="1xDeJPG_OmniSR.safetensors",
            ),
            architecture="OmniSR",
            scale=1,
            author="Philip Hofmann (Phips)",
            license="CC-BY-4.0",
            description="Ultra-lightweight 1x OmniSR artifact cleanup for high-speed runs.",
            source_url="https://huggingface.co/Phips/1xDeJPG_OmniSR",
            commercial_use=True,
        ),
        ModelSpec(
            key="exposure-compact",
            source=HuggingFaceSource(
                repo_id="Phips/1xExposureCorrection_compact",
                filename="1xExposureCorrection_compact.safetensors",
            ),
            architecture="Compact",
            scale=1,
            author="Philip Hofmann (Phips)",
            license="CC-BY-4.0",
            description="Lightweight 1x SRVGGNetCompact exposure and contrast correction.",
            source_url="https://huggingface.co/Phips/1xExposureCorrection_compact",
            commercial_use=True,
        ),
    ]
)


def resolve_model_path(
    model_ref: str,
    *,
    allow_unsafe_pickle: bool = False,
    registry: ModelRegistry = MODEL_REGISTRY,
) -> tuple[Path, ModelSpec | None]:
    """Resolve a registry key (downloading if needed) or validate a local weight file."""
    spec = registry.get(model_ref)
    if spec is not None:
        return spec.source.resolve(), spec

    path = Path(model_ref).expanduser()
    if not path.is_file():
        known = ", ".join(sorted(registry.keys()))
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


class FastGroupNorm(nn.Module):
    """Multi-SM GroupNorm replacement avoiding PyTorch's low-occupancy CUDA kernel."""

    def __init__(self, norm: nn.GroupNorm) -> None:
        super().__init__()
        self.num_groups = norm.num_groups
        self.num_channels = norm.num_channels
        self.eps = norm.eps
        self.weight = (
            nn.Parameter(norm.weight.detach().clone()) if norm.weight is not None else None
        )
        self.bias = nn.Parameter(norm.bias.detach().clone()) if norm.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        g = self.num_groups
        cpg = c // g
        if x.is_contiguous(memory_format=torch.channels_last):
            # Zero-copy view on NHWC layout: (N, H, W, C) -> (N, H, W, G, C//G)
            xhwc = x.permute(0, 2, 3, 1)
            xg = xhwc.reshape(n, h, w, g, cpg)
            var, mean = torch.var_mean(xg.float(), dim=(1, 2, 4), correction=0, keepdim=True)
            inv_std = torch.rsqrt(var + self.eps).to(x.dtype)
            y = ((xg - mean.to(x.dtype)) * inv_std).reshape(n, h, w, c)
            if self.weight is not None:
                y = y * self.weight.view(1, 1, 1, c)
            if self.bias is not None:
                y = y + self.bias.view(1, 1, 1, c)
            return y.permute(0, 3, 1, 2)

        xg = x.reshape(n, g, cpg, h, w)
        var, mean = torch.var_mean(xg.float(), dim=(2, 3, 4), correction=0, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps).to(x.dtype)
        y = ((xg - mean.to(x.dtype)) * inv_std).reshape(n, c, h, w)
        if self.weight is not None:
            y = y * self.weight.view(1, c, 1, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, c, 1, 1)
        return y


def _replace_group_norms(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, FastGroupNorm(child))
            replaced += 1
        else:
            replaced += _replace_group_norms(child)
    return replaced


ModelOptimizationPass = Callable[
    [nn.Module, DeviceInfo, torch.dtype],
    tuple[nn.Module, dict[str, Any]],
]


def replace_group_norm_pass(
    model: nn.Module,
    device_info: DeviceInfo,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, Any]]:
    """Replace nn.GroupNorm with FastGroupNorm on CUDA for higher SM utilization."""
    del dtype
    if device_info.backend != "cuda":
        return model, {}
    count = _replace_group_norms(model)
    if count == 0:
        return model, {}
    return model, {"label": f"fast_group_norm({count})", "replaced_group_norms": count}


def channels_last_pass(
    model: nn.Module,
    device_info: DeviceInfo,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, Any]]:
    """Convert half/bfloat16 CUDA models to channels_last (NHWC) for TensorCore kernels."""
    if device_info.backend != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return model, {"use_channels_last": False}
    converted = model.to(memory_format=torch.channels_last)
    return converted, {"label": "channels_last", "use_channels_last": True}


def compile_model_pass(
    model: nn.Module,
    device_info: DeviceInfo,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, Any]]:
    """Optionally compile model with torch.compile, gracefully falling back if unsupported."""
    del device_info, dtype
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model, {"compiled": False}
    try:
        compiled = compile_fn(model, mode="reduce-overhead")
    except Exception:
        return model, {"compiled": False}
    return compiled, {"label": "torch_compile", "compiled": True}


DEFAULT_OPTIMIZATION_PASSES: tuple[ModelOptimizationPass, ...] = (
    replace_group_norm_pass,
    channels_last_pass,
)


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
    tile_batch_size: int = 4,
) -> torch.Tensor:
    """Run a 1x NCHW model function over a batch of images, using batched feathered tiles."""
    if image.ndim != 4 or image.shape[0] < 1:
        raise ValueError(f"Expected NCHW tensor with batch>=1, got shape {tuple(image.shape)}")

    batch, channels, height, width = image.shape
    if tile_size == 0 or (height <= tile_size and width <= tile_size):
        padded, orig_h, orig_w = _pad_for_model(
            image, multiple_of=multiple_of, minimum=minimum
        )
        out = fn(padded)
        return out[:, :, :orig_h, :orig_w]

    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("tile_overlap must be >= 0 and strictly smaller than tile_size")

    is_cl = image.is_contiguous(memory_format=torch.channels_last)
    ys = _tile_starts(height, tile_size, tile_overlap)
    xs = _tile_starts(width, tile_size, tile_overlap)

    output_acc = torch.zeros(
        (batch, channels, height, width), dtype=torch.float32, device=image.device
    )
    weight_acc = torch.zeros((1, 1, height, width), dtype=torch.float32, device=image.device)

    y_masks = [
        _feather_mask_1d(
            min(y0 + tile_size, height) - y0,
            fade_left=tile_overlap if y0 > 0 else 0,
            fade_right=tile_overlap if min(y0 + tile_size, height) < height else 0,
            device=image.device,
        )
        for y0 in ys
    ]
    x_masks = [
        _feather_mask_1d(
            min(x0 + tile_size, width) - x0,
            fade_left=tile_overlap if x0 > 0 else 0,
            fade_right=tile_overlap if min(x0 + tile_size, width) < width else 0,
            device=image.device,
        )
        for x0 in xs
    ]

    # Group tiles by padded dimensions so same-sized tiles execute in one batched forward pass.
    grouped_tiles: dict[
        tuple[int, int], list[tuple[int, int, int, int, torch.Tensor, torch.Tensor]]
    ] = {}
    for yi, y0 in enumerate(ys):
        y1 = min(y0 + tile_size, height)
        wy = y_masks[yi]
        for xi, x0 in enumerate(xs):
            x1 = min(x0 + tile_size, width)
            wx = x_masks[xi]
            pad_key = _padded_shape(y1 - y0, x1 - x0, multiple_of=multiple_of, minimum=minimum)
            grouped_tiles.setdefault(pad_key, []).append((y0, y1, x0, x1, wy, wx))

    effective_tile_batch = max(1, tile_batch_size // batch)
    for tiles in grouped_tiles.values():
        for start in range(0, len(tiles), effective_tile_batch):
            chunk = tiles[start : start + effective_tile_batch]
            padded_patches: list[torch.Tensor] = []
            patch_dims: list[tuple[int, int]] = []
            for y0, y1, x0, x1, _, _ in chunk:
                patch = image[:, :, y0:y1, x0:x1]
                padded_patch, patch_h, patch_w = _pad_for_model(
                    patch, multiple_of=multiple_of, minimum=minimum
                )
                padded_patches.append(padded_patch)
                patch_dims.append((patch_h, patch_w))

            if len(padded_patches) == 1:
                batch_in = padded_patches[0]
            else:
                batch_in = torch.cat(padded_patches, dim=0)
            if is_cl:
                batch_in = batch_in.contiguous(memory_format=torch.channels_last)

            batch_out = fn(batch_in)
            outputs = (batch_out,) if len(chunk) == 1 else batch_out.split(batch, dim=0)

            for (y0, y1, x0, x1, wy, wx), (patch_h, patch_w), restored_patch in zip(
                chunk, patch_dims, outputs, strict=True
            ):
                cropped = restored_patch[:, :, :patch_h, :patch_w]
                weight_2d = (wy[:, None] * wx[None, :]).view(1, 1, patch_h, patch_w)
                output_acc[:, :, y0:y1, x0:x1] += cropped.to(torch.float32) * weight_2d
                weight_acc[:, :, y0:y1, x0:x1] += weight_2d

    return (output_acc / weight_acc.clamp_min(1e-6)).to(image.dtype)


def _is_oom_error(exc: RuntimeError) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


class Restorer:
    """Loads a 1x Spandrel restoration model and restores uint8 RGB video frames."""

    def __init__(
        self,
        model_ref: str,
        device_info: DeviceInfo,
        *,
        fp16: bool | None = None,
        optimize_gpu: bool = True,
        compile_model: bool = False,
        optimization_passes: Sequence[ModelOptimizationPass] | None = None,
        registry: ModelRegistry = MODEL_REGISTRY,
        allow_unsafe_pickle: bool = False,
    ) -> None:
        self.model_path, self.spec = resolve_model_path(
            model_ref,
            allow_unsafe_pickle=allow_unsafe_pickle,
            registry=registry,
        )
        self.device_info = device_info
        self.dtype = resolve_dtype(device_info, fp16)
        self.use_channels_last = False
        self.optimizations_applied: list[str] = []
        self._oom_fallback_tile_size: int | None = None
        self.last_tile_size: int | None = None

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
        model: nn.Module = descriptor.model.to(device=device_info.device, dtype=self.dtype)
        model.eval()

        if optimize_gpu and device_info.backend == "cuda":
            torch.backends.cudnn.benchmark = True
            self.optimizations_applied.append("cudnn_benchmark")
            passes = (
                list(optimization_passes)
                if optimization_passes is not None
                else list(DEFAULT_OPTIMIZATION_PASSES)
            )
            if compile_model:
                passes.append(compile_model_pass)
            for opt_pass in passes:
                model, meta = opt_pass(model, device_info, self.dtype)
                if meta.get("use_channels_last"):
                    self.use_channels_last = True
                label = meta.get("label")
                if isinstance(label, str):
                    self.optimizations_applied.append(label)
        elif compile_model:
            model, meta = compile_model_pass(model, device_info, self.dtype)
            if meta.get("label"):
                self.optimizations_applied.append(str(meta["label"]))

        self.model = model
        size_req = getattr(descriptor, "size_requirements", None)
        self.multiple_of = int(getattr(size_req, "multiple_of", 1) or 1)
        self.minimum = int(getattr(size_req, "minimum", 0) or 0)

    def _forward_model(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.use_channels_last:
            try:
                nhwc = tensor.contiguous(memory_format=torch.channels_last)
                return self.model(nhwc)
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    raise
                # Fall back cleanly if a custom architecture does not support channels_last.
                self.use_channels_last = False
                self.model = self.model.to(memory_format=torch.contiguous_format)
        return self.model(tensor.contiguous())

    def _resolve_effective_tile_size(
        self,
        requested_tile_size: int | None,
        width: int,
        height: int,
        batch_size: int,
    ) -> int:
        if self._oom_fallback_tile_size is not None and (
            requested_tile_size is None
            or requested_tile_size == 0
            or requested_tile_size > self._oom_fallback_tile_size
        ):
            return self._oom_fallback_tile_size
        if requested_tile_size is not None:
            return requested_tile_size
        return resolve_auto_tile_size(
            self.device_info,
            width,
            height,
            batch_size=batch_size,
            dtype=self.dtype,
        )

    def restore_batch(
        self,
        frames_rgb_u8: list[np.ndarray],
        *,
        ai_strength: float,
        tile_size: int | None,
        tile_overlap: int,
        tile_batch_size: int = 4,
    ) -> list[np.ndarray]:
        """Restore a batch of HWC uint8 RGB frames and blend with input by ai_strength."""
        if not frames_rgb_u8 or ai_strength <= 0.0:
            return frames_rgb_u8

        batch = len(frames_rgb_u8)
        if batch == 1:
            arr_u8 = np.require(frames_rgb_u8[0], requirements=["C", "W"])[None, ...]
        else:
            arr_u8 = np.stack(frames_rgb_u8, axis=0)

        height, width = arr_u8.shape[1], arr_u8.shape[2]
        cpu_tensor = torch.from_numpy(arr_u8)
        gpu_u8 = cpu_tensor.to(device=self.device_info.device, non_blocking=True)
        tensor = gpu_u8.permute(0, 3, 1, 2).to(dtype=self.dtype).div_(255.0)
        if self.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        effective_tile = self._resolve_effective_tile_size(tile_size, width, height, batch)
        effective_tile_batch = tile_batch_size

        with torch.inference_mode():
            try:
                restored = infer_tiled(
                    self._forward_model,
                    tensor,
                    tile_size=effective_tile,
                    tile_overlap=tile_overlap,
                    multiple_of=self.multiple_of,
                    minimum=self.minimum,
                    tile_batch_size=effective_tile_batch,
                )
            except RuntimeError as exc:
                if not _is_oom_error(exc) or (0 < effective_tile <= 512):
                    raise
                if self.device_info.backend == "cuda":
                    torch.cuda.empty_cache()
                fallback_tile = 512 if tile_overlap < 512 else max(tile_overlap + 64, 512)
                self._oom_fallback_tile_size = fallback_tile
                effective_tile = fallback_tile
                effective_tile_batch = 1
                restored = infer_tiled(
                    self._forward_model,
                    tensor,
                    tile_size=effective_tile,
                    tile_overlap=tile_overlap,
                    multiple_of=self.multiple_of,
                    minimum=self.minimum,
                    tile_batch_size=effective_tile_batch,
                )

            self.last_tile_size = effective_tile
            blended = (
                tensor * (1.0 - ai_strength) + restored * ai_strength
                if ai_strength < 1.0
                else restored
            )
            out_u8 = (
                blended.clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .permute(0, 2, 3, 1)
                .to(dtype=torch.uint8, device="cpu")
                .contiguous()
                .numpy()
            )

        out_u8 = np.ascontiguousarray(out_u8)
        return [np.ascontiguousarray(out_u8[i]) for i in range(batch)]

    def restore_frame(
        self,
        frame_rgb_u8: np.ndarray,
        *,
        ai_strength: float,
        tile_size: int | None,
        tile_overlap: int,
        tile_batch_size: int = 4,
    ) -> np.ndarray:
        """Restore a single HWC uint8 RGB frame and blend with the input by ai_strength."""
        return self.restore_batch(
            [frame_rgb_u8],
            ai_strength=ai_strength,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )[0]
