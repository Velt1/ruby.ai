# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in `ruby.ai`, please report it responsibly through GitHub Security Advisories or by contacting the repository maintainer directly rather than disclosing it in a public issue before a fix is available.

## Model Checkpoint Safety (`.safetensors` vs. Pickle)

`ruby.ai` enforces `.safetensors` weights for all built-in models to prevent arbitrary code execution risks associated with Python `pickle` deserialization.

- Local `.pth`, `.pt`, or `.ckpt` files are rejected by default.
- Loading pickle-based checkpoints requires passing `--allow-unsafe-pickle` explicitly and should only be done with weights from sources you fully trust.

## Subprocess Handling

External calls to `ffmpeg` and `ffprobe` always pass structured argument lists without `shell=True`, and user-provided numeric and enum options are validated before building filter graphs.
