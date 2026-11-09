# AGENTS.md — working notes for the next agent

## Rules (short)
- This is a personal, non-commercial project. Do not weaken tests/thresholds; if unavoidable, record numbers in `docs/DECISIONS.md`.
- Never edit the start prompt file in the repository root.
- All user documentation is Russian. Code, comments, commits: English.
- Keep files under ~400 lines. Do not print tensors/audio; print shape/min/max/NaN count only.
- Long jobs run through `tools\bg.ps1` (log in `logs/bg/<name>.log`, PID in `logs/bg/<name>.pid`). Never tie a training run to an interactive shell.
- Update `docs/STATUS.md` after every meaningful step; append to `docs/LOG.md`; tick `docs/PLAN.md`.
- Read order after context loss: `AGENTS.md` -> `docs/STATUS.md` -> `docs/PLAN.md` -> needed part of `docs/SPEC.md`.

## Environment (measured 2026-11-09)
- Windows 10/11 x64, RTX 3050 Laptop, 4096 MiB VRAM, driver 572.83.
- .NET SDK 10.0.403, 9.0.102, 8.0.423. Target: `net10.0-windows10.0.19041.0`.
- Python via `uv`; venv at `ml/.venv`. Torch CUDA 12.x.
- Data root: `E:\avis_data` (variable `AVIS_DATA`). E: had 369 GB free.
- GPU batch size 8 (4 GB VRAM rule).

## Commands
```powershell
# refresh PATH in a new shell
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')

tools\check_env.ps1
tools\build.ps1                 # Release x64 -> logs/build.log
tools\build.ps1 -Test           # build + xunit
tools\verify.ps1                # all checks -> artifacts/verify/<date>/report.md|json
tools\verify.ps1 -Only ml,frontend
tools\run.ps1                   # launches dist or runtime build
.\logs\...                      # check bg logs

# ML
cd ml
uv run pytest -q --tb=short
uv run python -m avis_ml.train --config configs/smoke.yaml
uv run python -m avis_ml.train --config configs/v1.yaml --resume
uv run python -m avis_ml.export --ckpt models/ckpt_last.pt --out ../models/v1
```

## Module map
- `ml/avis_ml/frontend.py` — FE-v1 reference (numpy streaming + torch batch), generates `shared/frontend/v1/weights.bin|frontend.json`.
- `ml/avis_ml/{scenarios,synth,mix,labels,losses,train,eval,export}.py` — data, model training, evaluation, ONNX export.
- `runtime/src/Avis.Core` — contracts (`IObjectSource`, `AnalysisFrame`), frontend (C# FFT), `ObjectParams`, `ObjectTracker`, `DspObjectSource`, `NeuralObjectSource`, settings model, JSONL.
- `runtime/src/Avis.Audio` — WASAPI loopback capture, resampler, SPSC ring (depends on NAudio only).
- `runtime/src/Avis.Render` — D3D11/Vortice renderer, HLSL, offscreen + PNG (depends only on Core).
- `runtime/src/Avis.App` — WinUI 3 shell, settings panel, overlay mode.
- `runtime/src/Avis.Cli` — `analyze|render|check|bench|parity|capture-test`.
- `shared/` — schemas, FE-v1 weights, golden vectors.
- `models/<version>/` — `model.onnx model.json frontend.json weights.bin`.

## Dependency boundaries
Core has no UI/render/NAudio. Render depends only on Core. App/Cli compose everything.
Work on one module: `dotnet build runtime/src/Avis.Core -c Release -p:Platform=x64`.

## Verification entry points
- `tools/verify.ps1 -Only frontend` — Python/C# parity on golden.
- `tools/verify.ps1 -Only scenarios` — CLI object checks S01–S13.
- `tools/verify.ps1 -Only render` — offscreen PNG pixel test + bench.
- `tools/verify.ps1 -Only model` — ONNX parity + latency.
- `avis-cli bench --what frontend|model|tracker|render`.
