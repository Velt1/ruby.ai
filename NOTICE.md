# Third-Party Notices & Attributions

`ruby.ai` is licensed under the MIT License (see `LICENSE`). It invokes external system tools and loads open-weight restoration models at runtime under their own respective licenses.

## Open-Weight Restoration Models (Downloaded at Runtime)

Model weights are not bundled in this repository and are downloaded on demand from Hugging Face / OpenModelDB when invoked by the user:

- **`1xDeH264_realplksr`** (`Phips/1xDeH264_realplksr`)
  - Author: Philip Hofmann (Phips)
  - Architecture: RealPLKSR (1x restoration)
  - License: Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
  - Source: https://huggingface.co/Phips/1xDeH264_realplksr

- **`1xDeNoise_realplksr_otf`** (`Phips/1xDeNoise_realplksr_otf`)
  - Author: Philip Hofmann (Phips)
  - Architecture: RealPLKSR (1x denoising)
  - License: Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
  - Source: https://huggingface.co/Phips/1xDeNoise_realplksr_otf

- **`1xDeJPG_realplksr_otf`** (`Phips/1xDeJPG_realplksr_otf`)
  - Author: Philip Hofmann (Phips)
  - Architecture: RealPLKSR (1x de-artifacting)
  - License: Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
  - Source: https://huggingface.co/Phips/1xDeJPG_realplksr_otf

When redistributing or deploying outputs or derivative configurations subject to `CC-BY-4.0`, ensure appropriate credit is given to Philip Hofmann (Phips) as required by the Creative Commons Attribution 4.0 license.

## Core Runtime Dependencies & External Tools

- **FFmpeg / FFprobe**: Invoked as external user-installed CLI subprocesses (not linked or redistributed with this repository). FFmpeg is licensed under LGPL v2.1+ or GPL v2+ depending on build configuration.
- **Spandrel** (`chaiNNer-org/spandrel`): MIT License
- **PyTorch / TorchVision**: BSD-3-Clause License
- **Safetensors**: Apache-2.0 License
- **Hugging Face Hub** (`huggingface-hub`): Apache-2.0 License
- **Typer & Rich**: MIT License
- **NumPy**: BSD-3-Clause License

## Trademark & Non-Affiliation Disclaimer

Topaz®, Topaz Labs®, Topaz Video AI®, and Proteus are trademarks or registered trademarks of Topaz Labs, LLC.

`ruby.ai` is an independent open-source project and is **not** affiliated with, endorsed by, sponsored by, or derived from Topaz Labs, LLC or any proprietary Topaz software or model weights. References to Topaz Video AI or Proteus are strictly factual and comparative to describe the general class of video restoration workflows.
