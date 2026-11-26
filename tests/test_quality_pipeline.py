"""Tests for Topaz-parity quality modules, 8GB VRAM RTX 3080 safeguards, and 16-bit pipe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ruby_ai.analyzer import analyze_video
from ruby_ai.config import EnhanceConfig, resolve_target_dimensions
from ruby_ai.device import DeviceInfo, resolve_auto_tiling_plan
from ruby_ai.ffmpeg import (
    build_decode_command,
    build_encode_command,
    build_encode_filter,
    probe_encoder_usable,
    require_binary,
)
from ruby_ai.models import Restorer
from ruby_ai.pipeline import enhance_video
from ruby_ai.postprocess import (
    DehaloProcessor,
    DetailRecoveryProcessor,
    WaveletColorFixProcessor,
    build_postprocess_pipeline,
)
from ruby_ai.temporal import TemporalStabilizer
from ruby_ai.tiling import TILE_BLENDING_STRATEGIES, infer_tiled


def _make_clip(path: Path, *, width: int = 192, height: int = 128, dur: float = 0.6) -> None:
    ffmpeg_bin = require_binary("ffmpeg")
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate=12:duration={dur}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_tiling_blending_strategies_partition_of_unity_1x_and_2x() -> None:
    torch.manual_seed(11)
    img = torch.rand((2, 3, 96, 128), dtype=torch.float32)

    for strategy_name in ("gaussian", "cosine", "linear"):
        assert strategy_name in TILE_BLENDING_STRATEGIES.keys()
        out_1x = infer_tiled(
            lambda x: x + 0.05,
            img,
            tile_size=48,
            tile_overlap=16,
            multiple_of=4,
            tile_batch_size=3,
            scale=1,
            blending=strategy_name,
        )
        assert out_1x.shape == img.shape
        assert (out_1x - (img + 0.05)).abs().max().item() < 1e-5

        out_2x = infer_tiled(
            lambda x: F.interpolate(x, scale_factor=2, mode="nearest"),
            img,
            tile_size=48,
            tile_overlap=12,
            multiple_of=4,
            tile_batch_size=2,
            scale=2,
            blending=strategy_name,
        )
        expected_2x = F.interpolate(img, scale_factor=2, mode="nearest")
        assert out_2x.shape == (2, 3, 192, 256)
        assert (out_2x - expected_2x).abs().max().item() < 1e-5


def test_postprocess_wavelet_color_detail_recovery_and_dehalo() -> None:
    torch.manual_seed(5)
    # Create textured reference with a dark patch on left and bright edge on right
    ref = torch.full((1, 3, 64, 64), 0.4, dtype=torch.float32)
    ref[:, :, :, :24] = 0.03  # deep shadow zone
    ref[:, 0, 16:48, 32:56] = 0.75
    # Add fine checkerboard texture in bright region
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    texture = ((yy + xx) % 2).float() * 0.06
    ref = (ref + texture.view(1, 1, 64, 64)).clamp(0.0, 1.0)

    # Simulate an AI output with a green/warm color shift + ringing halo
    shifted = ref.clone()
    shifted[:, 1] = (shifted[:, 1] + 0.12).clamp(0.0, 1.0)
    shifted[:, 2] = (shifted[:, 2] - 0.10).clamp(0.0, 1.0)
    shifted[:, :, 15, 32:56] = 0.98  # overshoot halo along edge

    color_fixed = WaveletColorFixProcessor(strength=1.0, radius=8.0).process(shifted, ref)
    # Chroma cast (+0.12 G / -0.10 B) is eliminated while luma anchor remains gentle
    gb_diff_fixed = (color_fixed[:, 1] - color_fixed[:, 2]).mean().item()
    gb_diff_ref = (ref[:, 1] - ref[:, 2]).mean().item()
    assert abs(gb_diff_fixed - gb_diff_ref) < 0.005
    color_locked = WaveletColorFixProcessor(strength=1.0, radius=8.0, luma_anchor=1.0).process(
        shifted, ref
    )
    assert (color_locked.mean(dim=(2, 3)) - ref.mean(dim=(2, 3))).abs().max().item() < 0.01

    dehaloed = DehaloProcessor(strength=0.8).process(shifted, ref)
    assert dehaloed[:, :, 15, 32:56].mean().item() < shifted[:, :, 15, 32:56].mean().item()

    # Smooth out fine texture to test DetailRecoveryProcessor
    smooth = F.avg_pool2d(ref, 3, stride=1, padding=1)
    recovered = DetailRecoveryProcessor(strength=0.5, coring=0.005, shadow_floor=0.08).process(
        smooth, ref
    )
    # Bright area texture restored, dark shadow area stays protected
    bright_diff = (recovered[:, :, 20:44, 34:54] - smooth[:, :, 20:44, 34:54]).abs().mean().item()
    dark_diff = (recovered[:, :, 20:44, 2:20] - smooth[:, :, 20:44, 2:20]).abs().mean().item()
    assert bright_diff > dark_diff * 2.0

    pipeline = build_postprocess_pipeline(keep_color=True, recover_details=0.25, dehalo=0.2)
    combined = pipeline.process(shifted, ref)
    assert combined.shape == ref.shape
    assert torch.isfinite(combined).all()


def test_temporal_stabilizer_tracks_motion_and_resets_on_scene_cut() -> None:
    torch.manual_seed(9)
    base = torch.zeros((1, 3, 96, 128), dtype=torch.float32)
    base[:, :, 24:72, 32:80] = 0.65

    # Frame 1 shifted 2px right relative to Frame 0, with opposite AI flicker noise
    f0_ref = base.clone()
    f1_ref = torch.roll(base, shifts=2, dims=3)
    f0_out = (f0_ref + 0.03).clamp(0.0, 1.0)
    f1_out = (f1_ref - 0.03).clamp(0.0, 1.0)

    stab = TemporalStabilizer(strength=0.8, scene_change_threshold=0.18)
    out0 = stab.process_batch(f0_out, f0_ref)
    assert torch.allclose(out0, f0_out)

    out1 = stab.process_batch(f1_out, f1_ref)
    # Stabilizer pulls f1_out gently toward warped f0_out without drifting wildly
    assert out1.mean().item() > f1_out.mean().item()

    # Hard scene cut (completely different bright frame): MUST NOT ghost previous frame!
    cut_ref = torch.full((1, 3, 96, 128), 0.92, dtype=torch.float32)
    cut_out = cut_ref.clone()
    out_cut = stab.process_batch(cut_out, cut_ref)
    assert torch.allclose(out_cut, cut_out, atol=1e-6)


def test_rtx_3080_8gb_auto_tiling_avoids_wddm_thrashing() -> None:
    rtx_3080_8gb = DeviceInfo(
        device=torch.device("cpu"),
        backend="cuda",
        name="NVIDIA GeForce RTX 3080 Laptop GPU",
        default_fp16=True,
        total_memory_bytes=8 * 1024**3,
    )
    plan_default = resolve_auto_tiling_plan(
        rtx_3080_8gb,
        1080,
        1920,
        batch_size=1,
        dtype=torch.float16,
        requested_tile_batch_size=4,
    )
    assert plan_default.tile_size == 512
    assert 1 <= plan_default.tile_batch_size <= 2

    plan_low_free = resolve_auto_tiling_plan(
        rtx_3080_8gb,
        1080,
        1920,
        batch_size=1,
        dtype=torch.float16,
        requested_tile_batch_size=4,
        free_memory_bytes=int(2.8 * 1024**3),
    )
    assert plan_low_free.tile_size == 384
    assert plan_low_free.tile_batch_size == 1


def test_restorer_progressive_oom_cascade_falls_back_cleanly(tmp_path: Path, monkeypatch) -> None:
    import spandrel

    class TinyIdentityNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    fake_weights = tmp_path / "dummy.safetensors"
    fake_weights.write_bytes(b"placeholder")
    fake_desc = spandrel.ImageModelDescriptor(
        TinyIdentityNet(),
        state_dict={},
        architecture="IdentityTest",
        purpose="Restoration",
        tags=[],
        supports_half=True,
        supports_bfloat16=False,
        scale=1,
        input_channels=3,
        output_channels=3,
    )
    monkeypatch.setattr(
        spandrel.ModelLoader,
        "load_from_file",
        lambda self, path: fake_desc,
    )

    cpu_dev = DeviceInfo(
        device=torch.device("cpu"),
        backend="cuda",
        name="Mock RTX 3080",
        default_fp16=False,
        total_memory_bytes=8 * 1024**3,
    )
    restorer = Restorer(
        str(fake_weights),
        DeviceInfo(device=torch.device("cpu"), backend="cpu", name="CPU", default_fp16=False),
    )
    restorer.device_info = cpu_dev

    attempts: list[tuple[int, int]] = []

    def flaky_forward(x: torch.Tensor) -> torch.Tensor:
        attempts.append((x.shape[0], x.shape[2]))
        if x.shape[2] > 400 or x.shape[0] > 1:
            raise RuntimeError("CUDA out of memory during test")
        return x

    monkeypatch.setattr(restorer, "_forward_model", flaky_forward)

    frame = np.full((640, 640, 3), 120, dtype=np.uint8)
    out = restorer.restore_batch(
        [frame],
        ai_strength=1.0,
        tile_size=0,
        tile_overlap=32,
        tile_batch_size=4,
    )
    assert len(out) == 1
    assert out[0].shape == (640, 640, 3)
    assert restorer.last_tile_size is not None and restorer.last_tile_size <= 384
    assert restorer.last_tile_batch_size == 1
    assert len(attempts) >= 2


def test_16bit_pipe_upscale_geometry_and_pro_codecs(tmp_path: Path) -> None:
    # 1. Upscale target dimensions
    assert resolve_target_dimensions(720, 1280, 2160, allow_upscale=True, model_scale=2) == (
        1440,
        2560,
    )
    assert resolve_target_dimensions(1080, 1920, 1080, allow_upscale=True, model_scale=2) == (
        1080,
        1920,
    )

    # 2. ProRes & HEVC command builders + runtime probe
    cfg_prores = EnhanceConfig(codec="prores_ks", pipe_bit_depth=16, export_target="master-4k")
    assert "format=yuv422p10le" in build_encode_filter(cfg_prores)
    from ruby_ai.ffmpeg import VideoInfo

    dummy_info = VideoInfo(
        path=Path("in.mp4"),
        width=1920,
        height=1080,
        display_width=1920,
        display_height=1080,
        rotation=0,
        fps_num=30,
        fps_den=1,
        fps_str="30/1",
        duration=1.0,
        frame_count=30,
        has_audio=False,
        pix_fmt="yuv420p",
        codec_name="h264",
    )
    dec_cmd = build_decode_command(dummy_info, cfg_prores, 1920, 1080, ffmpeg_bin="ffmpeg")
    assert "rgb48le" in dec_cmd
    enc_cmd = build_encode_command(
        dummy_info, Path("out.mov"), cfg_prores, 1920, 1080, ffmpeg_bin="ffmpeg"
    )
    assert "rgb48le" in enc_cmd
    assert "prores_ks" in enc_cmd

    ok_x264, _ = probe_encoder_usable("libx264")
    assert ok_x264 is True

    # 3. Real 16-bit rgb48le FFmpeg passthrough smoke run
    clip = tmp_path / "in.mp4"
    _make_clip(clip, width=160, height=120, dur=0.5)
    out_16 = tmp_path / "out_16.mp4"
    cfg_16 = EnhanceConfig.from_preset(
        "passthrough",
        max_short_edge=120,
        pipe_bit_depth=16,
        x264_preset="ultrafast",
    )
    res = enhance_video(clip, out_16, cfg_16, show_progress=False)
    assert out_16.is_file()
    assert res.pipe_bit_depth == 16
    assert res.frames_processed >= 5


def test_fast_analyzer_and_lazy_bridge_import(tmp_path: Path) -> None:
    clip = tmp_path / "sample.mp4"
    _make_clip(clip, width=320, height=240, dur=0.8)
    analysis = analyze_video(clip)
    assert 0.0 <= analysis.noise_score <= 1.0
    assert 0.0 <= analysis.sharpness_score <= 1.0
    assert 0.0 <= analysis.compression_score <= 1.0
    assert analysis.recommended_preset

    proc = subprocess.run(
        [sys.executable, "-c", "import ruby_ai.bridge, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
