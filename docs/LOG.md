# LOG (append only)

- 2026-11-09 P0 started. Verified Windows/x64, RTX 3050 Laptop 4 GB, .NET SDK 10.0.403 installed via winget, git 2.56 and uv 0.13.7 installed. Chose data disk E: (`E:\avis_data`, 369 GB free). Created repo skeleton and copied sections B–J verbatim into docs/SPEC.md.

## 2026-11-09 P1–P7 (сводка)
- P1: FE-v1 (numpy streaming + torch batch), golden, S01–S13, тесты. Коммит/тег p1.
- P2: процедурный миксер вместо Slakh/MUSDB (104 ГБ, нет места/времени), метки кадр×полоса, превью.
- P3: модель M-v1 (GRU 2×384), лоссы deep clustering + act/pitch/sal; smoke обучен (val 1.4154), v1 запущен в фоне.
- P4: C# Core/Audio/Render/Cli, паритет фронтенда 3.9e-6, JSONL, CLI analyze/render/check/bench/parity.
- P5: D3D11 offscreen + bloom + PNG; WinUI 3 App, swapchain, WASAPI loopback, настройки, bench/screenshot.
- P6: ObjectParams + DspObjectSource; DSP 9/13, обязательные 7/7 PASS.
- P7: export.py (opset 17, parity 1.7e-6), NeuralObjectSource, ObjectTracker, xUnit 7/7.
- Исправлено: presence всегда 0 (в трекер передавался LastPresence как override); члены трекера пулятся, LINQ убран из горячего пути.
- Добавлено: `--screenshot-at`, `--wav`, `--model` принимает путь или имя; tools/build|verify|publish|run.ps1.

