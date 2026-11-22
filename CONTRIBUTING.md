# Contributing to ruby.ai

Thank you for your interest in contributing to `ruby.ai`.

## Development Setup

```bash
uv sync
uv run ruff check src tests
uv run pytest -v
```

## Guidelines

- Keep modules focused, small, and typed (`src/ruby_ai/`).
- Add unit tests under `tests/` for any new configuration, geometry, tiling, or FFmpeg filter logic.
- Run `ruff check src tests` and `pytest` before opening a pull request.
- Never invoke `subprocess` with `shell=True`.

## Model Registry & Licensing Policy

- Built-in registry entries (`src/ruby_ai/models.py`) must use `.safetensors` weights (no pickled checkpoints).
- Any model added to the default registry must carry a clear, permissive or attribution-compatible open license (such as `MIT`, `Apache-2.0`, `BSD-3-Clause`, or `CC-BY-4.0`) with complete author, license, and source URL metadata reflected in `NOTICE.md`.
- Do not submit proprietary, unattributed, or non-commercial-only weights to the default registry.
