"""Motion-compensated GPU temporal stabilization with occlusion & scene-cut gating."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from ruby_ai.postprocess import _gaussian_blur_2d


@runtime_checkable
class SceneChangeDetector(Protocol):
    """Protocol for detecting hard scene cuts between consecutive frames."""

    def is_scene_change(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> bool:
        """Return True if `curr_frame` starts a new shot compared to `prev_frame` (1CHW)."""
        ...


@runtime_checkable
class MotionEstimator(Protocol):
    """Protocol for estimating dense 2D optical flow between consecutive frames."""

    def estimate(self, curr_frame: torch.Tensor, prev_frame: torch.Tensor) -> torch.Tensor:
        """Return (1, 2, H, W) pixel displacement field mapping curr coords into prev."""
        ...


def _rgb_to_luma(frame: torch.Tensor) -> torch.Tensor:
    return (
        0.2126 * frame[:, 0:1] + 0.7152 * frame[:, 1:2] + 0.0722 * frame[:, 2:3]
    ).float()


class LumaHistogramSceneDetector:
    """Fast GPU scene-cut gate combining downsampled luma L1, mean shift, and histogram diff."""

    def __init__(self, threshold: float = 0.18) -> None:
        self.threshold = float(threshold)

    def is_scene_change(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> bool:
        if prev_frame.shape != curr_frame.shape:
            return True

        prev_y = _rgb_to_luma(prev_frame)
        curr_y = _rgb_to_luma(curr_frame)

        prev_sm = F.interpolate(prev_y, size=(90, 160), mode="area")
        curr_sm = F.interpolate(curr_y, size=(90, 160), mode="area")

        l1_diff = (curr_sm - prev_sm).abs().mean().item()
        mean_shift = abs(curr_sm.mean().item() - prev_sm.mean().item())

        bins = 16
        p_idx = (prev_sm.reshape(-1).clamp(0.0, 0.9999) * bins).long()
        c_idx = (curr_sm.reshape(-1).clamp(0.0, 0.9999) * bins).long()
        p_hist = torch.bincount(p_idx, minlength=bins).float()
        c_hist = torch.bincount(c_idx, minlength=bins).float()
        p_hist = p_hist / p_hist.sum().clamp_min(1.0)
        c_hist = c_hist / c_hist.sum().clamp_min(1.0)
        hist_diff = 0.5 * (p_hist - c_hist).abs().sum().item()

        score = 0.55 * l1_diff + 0.35 * hist_diff + 0.10 * mean_shift
        return score >= self.threshold


def warp_tensor(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp NCHW `tensor` using pixel displacement `flow` (N, 2, H, W) via bilinear sampling."""
    _, _, h, w = tensor.shape
    device = tensor.device
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    norm_dx = flow[:, 0] * (2.0 / max(1, w - 1))
    norm_dy = flow[:, 1] * (2.0 / max(1, h - 1))
    grid = torch.stack((grid_x.unsqueeze(0) + norm_dx, grid_y.unsqueeze(0) + norm_dy), dim=-1)
    warped = F.grid_sample(
        tensor.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.to(tensor.dtype)


class PyTorchLucasKanadeFlow:
    """Vectorized coarse-to-fine multi-scale Lucas-Kanade optical flow in pure PyTorch."""

    def __init__(
        self,
        *,
        levels: int = 3,
        window_size: int = 5,
        iterations: int = 2,
    ) -> None:
        self.levels = max(1, int(levels))
        self.window_size = max(3, int(window_size))
        self.iterations = max(1, int(iterations))

    def _refine_level(
        self,
        curr_y: torch.Tensor,
        prev_y: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        pad = self.window_size // 2
        # Central finite-difference spatial gradients on curr_y
        curr_p = F.pad(curr_y, (1, 1, 1, 1), mode="replicate")
        ix = 0.5 * (curr_p[:, :, 1:-1, 2:] - curr_p[:, :, 1:-1, :-2])
        iy = 0.5 * (curr_p[:, :, 2:, 1:-1] - curr_p[:, :, :-2, 1:-1])

        s_xx = F.avg_pool2d(ix * ix, self.window_size, stride=1, padding=pad)
        s_yy = F.avg_pool2d(iy * iy, self.window_size, stride=1, padding=pad)
        s_xy = F.avg_pool2d(ix * iy, self.window_size, stride=1, padding=pad)

        # Ridge damping stabilizes flat / textureless sky or wall patches.
        damping = 1e-3
        a11 = s_xx + damping
        a22 = s_yy + damping
        det = (a11 * a22 - s_xy * s_xy).clamp_min(1e-6)

        for _ in range(self.iterations):
            warped_prev = warp_tensor(prev_y, flow)
            it = warped_prev - curr_y
            b1 = -F.avg_pool2d(ix * it, self.window_size, stride=1, padding=pad)
            b2 = -F.avg_pool2d(iy * it, self.window_size, stride=1, padding=pad)

            du = (a22 * b1 - s_xy * b2) / det
            dv = (a11 * b2 - s_xy * b1) / det
            delta = torch.cat((du, dv), dim=1).clamp(-4.0, 4.0)
            flow = flow + delta

        return flow

    def estimate(self, curr_frame: torch.Tensor, prev_frame: torch.Tensor) -> torch.Tensor:
        _, _, h, w = curr_frame.shape
        curr_y = _rgb_to_luma(curr_frame)
        prev_y = _rgb_to_luma(prev_frame)

        # Start pyramid at 1/4 resolution for high speed and wide search range.
        base_h = max(16, h // 4)
        base_w = max(16, w // 4)
        curr_q = F.interpolate(curr_y, size=(base_h, base_w), mode="area")
        prev_q = F.interpolate(prev_y, size=(base_h, base_w), mode="area")

        pyr_c = [curr_q]
        pyr_p = [prev_q]
        for _ in range(self.levels - 1):
            ch, cw = pyr_c[-1].shape[2], pyr_c[-1].shape[3]
            if min(ch, cw) < 16:
                break
            pyr_c.append(F.avg_pool2d(pyr_c[-1], 2))
            pyr_p.append(F.avg_pool2d(pyr_p[-1], 2))

        coarsest = pyr_c[-1]
        flow = torch.zeros(
            (coarsest.shape[0], 2, coarsest.shape[2], coarsest.shape[3]),
            dtype=torch.float32,
            device=curr_frame.device,
        )

        for lvl in range(len(pyr_c) - 1, -1, -1):
            lc = pyr_c[lvl]
            lp = pyr_p[lvl]
            if flow.shape[2:] != lc.shape[2:]:
                scale_x = lc.shape[3] / max(1, flow.shape[3])
                scale_y = lc.shape[2] / max(1, flow.shape[2])
                flow = F.interpolate(flow, size=lc.shape[2:], mode="bilinear", align_corners=False)
                flow = torch.stack((flow[:, 0] * scale_x, flow[:, 1] * scale_y), dim=1)
            flow = self._refine_level(lc, lp, flow)

        # Upsample flow cleanly back to full frame resolution.
        if flow.shape[2:] != (h, w):
            scale_x = w / max(1, flow.shape[3])
            scale_y = h / max(1, flow.shape[2])
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow = torch.stack((flow[:, 0] * scale_x, flow[:, 1] * scale_y), dim=1)

        return flow


class TemporalStabilizer:
    """Recurrent flow-warped temporal stabilizer preventing AI shimmer without ghosting."""

    def __init__(
        self,
        *,
        strength: float = 0.25,
        scene_change_threshold: float = 0.18,
        occlusion_sigma: float = 0.045,
        max_delta: float = 0.075,
        scene_detector: SceneChangeDetector | None = None,
        motion_estimator: MotionEstimator | None = None,
    ) -> None:
        self.strength = float(strength)
        self.scene_change_threshold = float(scene_change_threshold)
        self.occlusion_sigma = float(occlusion_sigma)
        self.max_delta = float(max_delta)
        self.scene_detector = scene_detector or LumaHistogramSceneDetector(
            threshold=self.scene_change_threshold
        )
        self.motion_estimator = motion_estimator or PyTorchLucasKanadeFlow()
        self._prev_ref: torch.Tensor | None = None
        self._prev_out: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear recurrent history (e.g. between unrelated clips)."""
        self._prev_ref = None
        self._prev_out = None

    def process_batch(
        self,
        restored: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Stabilize an NCHW batch sequentially along the batch/time dimension."""
        if self.strength <= 0.0 or restored.shape[0] == 0:
            return restored

        orig_dtype = restored.dtype
        outputs: list[torch.Tensor] = []

        for idx in range(restored.shape[0]):
            curr_out = restored[idx : idx + 1].float()
            curr_ref = reference[idx : idx + 1].float()

            if (
                self._prev_ref is None
                or self._prev_out is None
                or self._prev_ref.shape != curr_ref.shape
                or self.scene_detector.is_scene_change(self._prev_ref, curr_ref)
            ):
                stabilized = curr_out
            else:
                flow = self.motion_estimator.estimate(curr_ref, self._prev_ref)
                warped_prev_ref = warp_tensor(self._prev_ref, flow)
                warped_prev_out = warp_tensor(self._prev_out, flow)

                warp_err = (warped_prev_ref - curr_ref).abs().mean(dim=1, keepdim=True)
                warp_err_smooth = _gaussian_blur_2d(warp_err, sigma=1.5, kernel_size=5)
                confidence = torch.exp(
                    -((warp_err_smooth / max(1e-4, self.occlusion_sigma)) ** 2)
                )

                blend_w = (self.strength * 0.55) * confidence
                diff = (warped_prev_out - curr_out).clamp(-self.max_delta, self.max_delta)
                stabilized = (curr_out + blend_w * diff).clamp(0.0, 1.0)

            self._prev_ref = curr_ref.detach().clone()
            self._prev_out = stabilized.detach().clone()
            outputs.append(stabilized.to(orig_dtype))

        return torch.cat(outputs, dim=0)
