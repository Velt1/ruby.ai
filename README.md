# ruby.ai

Independent open-source 4K-to-1080p video restoration and supersampling CLI tailored for Instagram Reels and TikTok.

Inspired by commercial Proteus-style video restoration workflows (such as Topaz Video AI), `ruby.ai` combines temporal pre-denoising, high-quality Lanczos supersampling, tiled 1x open-weight neural restoration (`spandrel` + OpenModelDB models), adaptive edge sharpening, and compression-resilient BT.709 H.264 encoding via FFmpeg.

## How the 5-Stage Pipeline Works

1. **Temporal Pre-Denoise (`atadenoise`):** Reduces frame-to-frame sensor flicker and fine high-ISO noise before spatial processing so social media re-encoders spend bitrate on real detail rather than random grain.
2. **Supersampling (`4K -> 1080p` via `lanczos`):** Downsamples 4K source footage cleanly to 1080p short-edge (`1080x1920` portrait or `1920x1080` landscape) with rotation-aware geometry, preserving 4K sharpness while reducing downstream GPU compute by 4x.
3. **1x Neural Restoration + Blend (`spandrel` + OpenModelDB):** Runs tiled 1x restoration weights (`RealPLKSR`) with feathered tile blending and mixes the restored frame back with the clean supersampled frame (`ai_strength`, default `0.65`) to preserve natural skin texture and avoid over-smoothed plastic artifacts.
4. **Adaptive Detail Polish (`cas` + subtle temporal grain):** Applies Contrast Adaptive Sharpening without halos and optional subtle temporal grain to prevent color banding in dark gradients.
5. **Social-Media Export (`libx264` / NVENC / VideoToolbox):** Outputs BT.709 `yuv420p` MP4 with `+faststart` and AAC audio, plus a built-in social compression simulator (`benchmark`) to inspect results *after* aggressive ~3.5 Mbps re-encoding.

## Requirements

- Python 3.11–3.13 (`3.12` recommended via `uv`)
- [`uv`](https://docs.astral.sh/uv/) package manager
- `ffmpeg` and `ffprobe` available on `PATH`
- **GPU:** NVIDIA CUDA GPU recommended (e.g. RTX 4070 / RTX Laptop GPU); Apple Silicon (`mps`) and CPU fallback are also supported.

## Quickstart

Install dependencies:

```bash
uv sync
```

Check your environment (PyTorch backend, FFmpeg, encoders):

```bash
uv run ruby-ai doctor
```

List built-in verified restoration models and their licenses:

```bash
uv run ruby-ai models
```

Enhance a 4K clip to a 1080p social-ready MP4:

```bash
uv run ruby-ai enhance input_4k.mp4 -o output_1080p.mp4 --preset crisp
```

Run an A/B social-compression benchmark (creates baseline Lanczos, `ruby.ai`, and simulated ~3.5 Mbps re-encoded comparisons plus a side-by-side PNG):

```bash
uv run ruby-ai benchmark input_4k.mp4 --start 2 --duration 5 --reference topaz_export.mp4
```

## Presets

| Preset | Model | Temporal Denoise | AI Strength | CAS Sharpen | Grain | Use Case |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `crisp` | `de-h264` | `0.03` | `0.65` | `0.25` | `2` | Default balanced clarity for clean 4K clips |
| `natural` | `de-h264` | `0.02` | `0.45` | `0.15` | `2` | Subtle restoration preserving organic texture |
| `low-light` | `denoise` | `0.05` | `0.75` | `0.20` | `3` | High-ISO / noisy indoor or night footage |
| `passthrough` | `de-h264` | `0.02` | `0.00` | `0.15` | `0` | Pure FFmpeg temporal denoise + supersampling (no neural pass) |

Any preset parameter can be overridden on the CLI (`--ai-strength`, `--temporal-denoise`, `--sharpen`, `--grain`, `--model`, `--crf`, `--codec`).

## License & Model Attributions

- The `ruby.ai` source code is licensed under the [MIT License](LICENSE).
- Default 1x neural restoration weights (`1xDeH264_realplksr`, `1xDeNoise_realplksr_otf`, `1xDeJPG_realplksr_otf`) are authored by **Philip Hofmann (Phips)** and licensed separately under **CC-BY-4.0**. They are downloaded on demand from Hugging Face / OpenModelDB and are not redistributed inside this repository. See [NOTICE.md](NOTICE.md) for full third-party license details.

## Trademarks & Legal Disclaimer

Topaz®, Topaz Labs®, Topaz Video AI®, and Proteus are trademarks or registered trademarks of Topaz Labs, LLC. `ruby.ai` is an independent open-source project and is **not** affiliated with, endorsed by, sponsored by, or derived from Topaz Labs, LLC or any proprietary Topaz software or model weights. All references to Topaz Video AI or Proteus are strictly factual and comparative.
