"""Open-weight restoration model registry, safe loading, multi-scale tiling, and GPU inference."""

from __future__ import annotations

import hashlib
import inspect
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
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from ruby_ai.device import (
    DeviceInfo,
    query_cuda_memory_info,
    resolve_auto_tiling_plan,
    resolve_dtype,
)
from ruby_ai.postprocess import build_postprocess_pipeline
from ruby_ai.temporal import TemporalStabilizer
from ruby_ai.tiling import (
    TileBlendingStrategy,
    _pad_for_model,
    _padded_shape,
    _tile_starts,
    infer_tiled,
)

__all__ = [
    "ALLOWED_MODEL_LICENSES",
    "DEFAULT_OPTIMIZATION_PASSES",
    "DirectUrlSource",
    "FastGroupNorm",
    "HuggingFaceSource",
    "MODEL_REGISTRY",
    "ModelRegistry",
    "ModelSource",
    "ModelSpec",
    "PICKLE_SUFFIXES",
    "Restorer",
    "_pad_for_model",
    "_padded_shape",
    "_tile_starts",
    "channels_last_pass",
    "compile_model_pass",
    "get_model_cache_dir",
    "get_model_spec_scale",
    "infer_tiled",
    "replace_group_norm_pass",
    "resolve_model_path",
]

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

ModelDownloadProgressCallback = Callable[[int, int | None], None]
ModelStatusCallback = Callable[[str, str, str | None], None]


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

    def resolve(
        self,
        *,
        on_status: ModelStatusCallback | None = None,
        on_progress: ModelDownloadProgressCallback | None = None,
    ) -> Path:
        del on_progress
        cached = try_to_load_from_cache(repo_id=self.repo_id, filename=self.filename)
        if isinstance(cached, str) and Path(cached).is_file():
            if on_status is not None:
                on_status("model", f"Lade gecachtes Modell '{self.filename}'…", None)
            return Path(cached)

        if on_status is not None:
            on_status(
                "model",
                f"Lade KI-Modell '{self.filename}' von Hugging Face…",
                f"Repo: {self.repo_id} (einmaliger Download)",
            )
        return Path(hf_hub_download(repo_id=self.repo_id, filename=self.filename))


@dataclass(frozen=True, slots=True)
class DirectUrlSource:
    """Model weight source downloaded via HTTPS and cached locally."""

    url: str
    filename: str
    sha256: str | None = None
    source_kind: str = "url"

    def resolve(
        self,
        *,
        on_status: ModelStatusCallback | None = None,
        on_progress: ModelDownloadProgressCallback | None = None,
    ) -> Path:
        cache_dir = get_model_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / self.filename
        expected = self.sha256.lower() if self.sha256 else None

        if dest.is_file() and dest.stat().st_size > 0:
            if expected is None or _sha256_file(dest) == expected:
                if on_status is not None:
                    on_status("model", f"Lade gecachtes Modell '{self.filename}'…", None)
                return dest
            dest.unlink(missing_ok=True)

        if on_status is not None:
            on_status(
                "model",
                f"Lade KI-Modell '{self.filename}' herunter…",
                self.url,
            )

        part = cache_dir / f"{self.filename}.{os.getpid()}.part"
        req = urllib.request.Request(self.url, headers={"User-Agent": "ruby-ai/0.2"})
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
                headers = getattr(resp, "headers", None)
                raw_len = headers.get("Content-Length") if headers else None
                total_bytes = int(raw_len) if raw_len and str(raw_len).isdigit() else None
                downloaded = 0
                last_emit = 0
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if on_progress is not None and (
                        downloaded - last_emit >= 512 * 1024
                        or (total_bytes is not None and downloaded >= total_bytes)
                    ):
                        last_emit = downloaded
                        on_progress(downloaded, total_bytes)
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


def _resolve_source(
    source: ModelSource,
    *,
    on_status: ModelStatusCallback | None = None,
    on_progress: ModelDownloadProgressCallback | None = None,
) -> Path:
    try:
        params = inspect.signature(source.resolve).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {}
    if "on_status" in params and on_status is not None:
        kwargs["on_status"] = on_status
    if "on_progress" in params and on_progress is not None:
        kwargs["on_progress"] = on_progress
    return source.resolve(**kwargs)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata and attribution for a verified open-weight restoration or upscaling model."""

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
        ModelSpec(
            key="upscale-2x-span",
            source=HuggingFaceSource(
                repo_id="Phips/2xNomosUni_span_multijpg_ldl",
                filename="2xNomosUni_span_multijpg_ldl.safetensors",
            ),
            architecture="SPAN",
            scale=2,
            author="Philip Hofmann (Phips)",
            license="CC-BY-4.0",
            description=(
                "Fast 2x SPAN universal upscaler preserving depth-of-field and fine textures."
            ),
            source_url="https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl",
            commercial_use=True,
        ),
    ]
)


def get_model_spec_scale(
    model_ref: str,
    *,
    registry: ModelRegistry = MODEL_REGISTRY,
) -> int:
    """Return declared scale factor for a registered model key, defaulting to 1."""
    spec = registry.get(model_ref)
    return int(spec.scale) if spec is not None else 1


def resolve_model_path(
    model_ref: str,
    *,
    allow_unsafe_pickle: bool = False,
    registry: ModelRegistry = MODEL_REGISTRY,
    on_status: ModelStatusCallback | None = None,
    on_progress: ModelDownloadProgressCallback | None = None,
) -> tuple[Path, ModelSpec | None]:
    """Resolve a registry key (downloading if needed) or validate a local weight file."""
    spec = registry.get(model_ref)
    if spec is not None:
        return _resolve_source(spec.source, on_status=on_status, on_progress=on_progress), spec

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


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in msg or "alloc" in msg


class Restorer:
    """Loads a 1x/2x/4x Spandrel model and restores uint8 or uint16 RGB video frames."""

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
        on_status: ModelStatusCallback | None = None,
        on_download_progress: ModelDownloadProgressCallback | None = None,
    ) -> None:
        self.model_path, self.spec = resolve_model_path(
            model_ref,
            allow_unsafe_pickle=allow_unsafe_pickle,
            registry=registry,
            on_status=on_status,
            on_progress=on_download_progress,
        )
        self.device_info = device_info
        self.dtype = resolve_dtype(device_info, fp16)
        self.use_channels_last = False
        self.optimizations_applied: list[str] = []
        self._oom_fallback_tile_size: int | None = None
        self._oom_fallback_tile_batch_size: int | None = None
        self.last_tile_size: int | None = None
        self.last_tile_batch_size: int = 1
        self._temporal_stabilizer: TemporalStabilizer | None = None
        self._temporal_key: tuple[float, float] | None = None

        if on_status is not None:
            on_status(
                "model_load",
                f"Initialisiere Modell auf {device_info.name}…",
                self.model_path.name,
            )

        descriptor = spandrel.ModelLoader(device=torch.device("cpu")).load_from_file(
            self.model_path
        )
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise TypeError(
                f"Expected an ImageModelDescriptor, got {type(descriptor).__name__}"
            )
        if descriptor.scale not in {1, 2, 4}:
            raise ValueError(
                f"Unsupported model scale={descriptor.scale}x (supported: 1x, 2x, 4x)."
            )

        self.descriptor = descriptor
        self.scale = int(descriptor.scale)
        model: nn.Module = descriptor.model.to(device=device_info.device, dtype=self.dtype)
        model.eval()

        if optimize_gpu and device_info.backend == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.optimizations_applied.append("tf32")

            # Avoid cuDNN autotuner workspace spikes & first-frame stalls on 8 GB GPUs.
            mem_info = query_cuda_memory_info(device_info)
            free_gib = (mem_info[0] / (1024**3)) if mem_info is not None else None
            total_gib = (
                device_info.total_memory_bytes / (1024**3)
                if device_info.total_memory_bytes is not None
                else 0.0
            )
            allow_bench = (free_gib >= 7.5) if free_gib is not None else (total_gib >= 11.5)
            torch.backends.cudnn.benchmark = allow_bench
            if allow_bench:
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

    def reset_temporal(self) -> None:
        """Reset recurrent temporal state (e.g. before a pipeline retry)."""
        if self._temporal_stabilizer is not None:
            self._temporal_stabilizer.reset()

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

    def _build_oom_candidates(
        self,
        initial_tile: int,
        initial_tb: int,
        tile_overlap: int,
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = [(initial_tile, max(1, initial_tb))]
        fallbacks = [(512, 2), (512, 1), (384, 1), (256, 1)]
        for ft, ftb in fallbacks:
            if ft <= tile_overlap:
                continue
            if initial_tile == 0 or ft < initial_tile or (ft == initial_tile and ftb < initial_tb):
                if (ft, ftb) not in candidates:
                    candidates.append((ft, ftb))
        return candidates

    def _infer_with_oom_cascade(
        self,
        tensor: torch.Tensor,
        *,
        initial_tile: int,
        initial_tb: int,
        tile_overlap: int,
        blending: str | TileBlendingStrategy,
    ) -> tuple[torch.Tensor, int, int]:
        candidates = self._build_oom_candidates(initial_tile, initial_tb, tile_overlap)
        last_exc: RuntimeError | None = None

        for idx, (cand_tile, cand_tb) in enumerate(candidates):
            try:
                out = infer_tiled(
                    self._forward_model,
                    tensor,
                    tile_size=cand_tile,
                    tile_overlap=tile_overlap,
                    multiple_of=self.multiple_of,
                    minimum=self.minimum,
                    tile_batch_size=cand_tb,
                    scale=self.scale,
                    blending=blending,
                )
                if idx > 0:
                    self._oom_fallback_tile_size = cand_tile
                    self._oom_fallback_tile_batch_size = cand_tb
                return out, cand_tile, cand_tb
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                last_exc = exc
                if self.device_info.backend == "cuda":
                    torch.backends.cudnn.benchmark = False
                    torch.cuda.empty_cache()

        # If a multi-frame batch still OOMs, split frame-by-frame at the smallest safe tile.
        if tensor.shape[0] > 1:
            safe_tile = candidates[-1][0]
            slices: list[torch.Tensor] = []
            for frame_idx in range(tensor.shape[0]):
                sub = infer_tiled(
                    self._forward_model,
                    tensor[frame_idx : frame_idx + 1],
                    tile_size=safe_tile,
                    tile_overlap=tile_overlap,
                    multiple_of=self.multiple_of,
                    minimum=self.minimum,
                    tile_batch_size=1,
                    scale=self.scale,
                    blending=blending,
                )
                slices.append(sub)
            self._oom_fallback_tile_size = safe_tile
            self._oom_fallback_tile_batch_size = 1
            return torch.cat(slices, dim=0), safe_tile, 1

        assert last_exc is not None
        raise last_exc

    def _get_temporal_stabilizer(
        self,
        temporal_consistency: float,
        scene_change_threshold: float,
    ) -> TemporalStabilizer | None:
        if temporal_consistency <= 0.0:
            return None
        key = (round(temporal_consistency, 4), round(scene_change_threshold, 4))
        if self._temporal_stabilizer is None or self._temporal_key != key:
            self._temporal_stabilizer = TemporalStabilizer(
                strength=temporal_consistency,
                scene_change_threshold=scene_change_threshold,
            )
            self._temporal_key = key
        return self._temporal_stabilizer

    def restore_batch(
        self,
        frames_rgb: list[np.ndarray],
        *,
        ai_strength: float,
        tile_size: int | None,
        tile_overlap: int,
        tile_batch_size: int = 4,
        tile_blending: str | TileBlendingStrategy = "gaussian",
        keep_color: bool = True,
        recover_details: float = 0.0,
        dehalo: float = 0.0,
        temporal_consistency: float = 0.0,
        scene_change_threshold: float = 0.18,
    ) -> list[np.ndarray]:
        """Restore a batch of HWC uint8/uint16 RGB frames with post-processing & temporal gate."""
        if not frames_rgb:
            return frames_rgb
        if ai_strength <= 0.0 and self.scale == 1:
            return frames_rgb

        is_u16 = frames_rgb[0].dtype == np.uint16
        batch = len(frames_rgb)
        if batch == 1:
            arr = np.require(frames_rgb[0], requirements=["C"])[None, ...]
        else:
            arr = np.stack(frames_rgb, axis=0)

        height, width = arr.shape[1], arr.shape[2]
        scale_max = 65535.0 if is_u16 else 255.0
        cpu_arr = arr.astype(np.int32, copy=False) if is_u16 else arr
        cpu_tensor = torch.from_numpy(cpu_arr)
        if self.device_info.backend == "cuda" and not cpu_tensor.is_pinned():
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError:
                pass

        gpu_raw = cpu_tensor.to(device=self.device_info.device, non_blocking=True)
        tensor = gpu_raw.permute(0, 3, 1, 2).to(dtype=self.dtype).div_(scale_max)
        if self.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        mem_info = query_cuda_memory_info(self.device_info) if tile_size is None else None
        free_bytes = mem_info[0] if mem_info is not None else None
        plan = resolve_auto_tiling_plan(
            self.device_info,
            width,
            height,
            batch_size=batch,
            dtype=self.dtype,
            model_scale=self.scale,
            requested_tile_size=tile_size,
            requested_tile_batch_size=tile_batch_size,
            free_memory_bytes=free_bytes,
        )

        effective_tile = plan.tile_size
        effective_tb = plan.tile_batch_size
        if self._oom_fallback_tile_size is not None and (
            effective_tile == 0 or effective_tile > self._oom_fallback_tile_size
        ):
            effective_tile = self._oom_fallback_tile_size
        if self._oom_fallback_tile_batch_size is not None:
            effective_tb = min(effective_tb, self._oom_fallback_tile_batch_size)

        with torch.inference_mode():
            restored, used_tile, used_tb = self._infer_with_oom_cascade(
                tensor,
                initial_tile=effective_tile,
                initial_tb=effective_tb,
                tile_overlap=tile_overlap,
                blending=tile_blending,
            )
            self.last_tile_size = used_tile
            self.last_tile_batch_size = used_tb

            if self.scale == 1:
                reference = tensor
            else:
                reference = (
                    F.interpolate(
                        tensor.float(),
                        size=(restored.shape[2], restored.shape[3]),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .clamp(0.0, 1.0)
                    .to(restored.dtype)
                )

            post_pipeline = build_postprocess_pipeline(
                keep_color=keep_color,
                recover_details=recover_details,
                dehalo=dehalo,
            )
            if post_pipeline:
                restored = post_pipeline.process(restored, reference)

            if ai_strength < 1.0:
                blended = reference * (1.0 - ai_strength) + restored * ai_strength
            else:
                blended = restored

            stabilizer = self._get_temporal_stabilizer(
                temporal_consistency, scene_change_threshold
            )
            if stabilizer is not None:
                blended = stabilizer.process_batch(blended, reference)

            scaled = blended.clamp(0.0, 1.0).mul(scale_max).round().permute(0, 2, 3, 1)
            if is_u16:
                out_np = (
                    scaled.to(dtype=torch.int32, device="cpu")
                    .contiguous()
                    .numpy()
                    .astype("<u2", copy=False)
                )
            else:
                out_np = scaled.to(dtype=torch.uint8, device="cpu").contiguous().numpy()

        out_np = np.ascontiguousarray(out_np)
        return [np.ascontiguousarray(out_np[i]) for i in range(batch)]

    def restore_frame(
        self,
        frame_rgb: np.ndarray,
        *,
        ai_strength: float,
        tile_size: int | None,
        tile_overlap: int,
        tile_batch_size: int = 4,
        tile_blending: str | TileBlendingStrategy = "gaussian",
        keep_color: bool = True,
        recover_details: float = 0.0,
        dehalo: float = 0.0,
        temporal_consistency: float = 0.0,
        scene_change_threshold: float = 0.18,
    ) -> np.ndarray:
        """Restore a single HWC uint8/uint16 RGB frame and blend with the input by ai_strength."""
        return self.restore_batch(
            [frame_rgb],
            ai_strength=ai_strength,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            tile_blending=tile_blending,
            keep_color=keep_color,
            recover_details=recover_details,
            dehalo=dehalo,
            temporal_consistency=temporal_consistency,
            scene_change_threshold=scene_change_threshold,
        )[0]
