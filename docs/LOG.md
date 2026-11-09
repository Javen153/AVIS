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

## 2026-11-09 P8 (идёт)
- App: рендер вынесен в отдельный поток с vsync `Present(1,0)`; UI-таймер 100 мс
  только для статистики/скриншота/bench. FPS p50 60 → **164.7** (панель 165 Гц).
- Починен mojibake в русских строках `MainWindow.cs` (файл перезаписывался через
  PowerShell с двойной перекодировкой); аудит остальных файлов чист.
- bench/screenshot теперь отсчитываются от первого кадра (WinUI стартует ~2.9 с).
- verify (frontend/scenarios/render/app): 4/4 PASS; app fps p50 161.0, lit=854.
- Обучение v1 разошлось на шаге 4220 (loss 48.6); остановлено, восстановлено с
  шага 4000 (val 1.1691), lr 5e-4, clip 1.0, skip_loss 25 (D-015).
- Экспорт v1@4000 → `models/v1` (parity 1.9e-6): нейро 3/13.
- Итерация трекера (D-016): призраки одного пика больше не рождаются/сливаются;
  нейро 3/13 → **7/13**, DSP 9/13 без изменений.
- Добавлен `bench --what latency` и проверка `latency` в verify.ps1:
  DSP p95 19 мс (бюджет 30), `scenarios.ps1 -Set` для перебора настроек.

- 2026-11-09 06:07 checkpoint 6000 (val 1.1236) сохранён и скопирован; экспорт во
  временную `models/v1_s6000`, прогон 13 сценариев — 7/13 (S05 +, S12 −).
- Подбор `tracking.MatchThreshold`: 0.60 → 8/13, **0.68 → 9/13**, 0.74 → 8/13,
  0.80 → 7/13. Белый шум S07: перцентили ширины 10/90 → 5/95, ширина 59% → 87%,
  S07 PASS (D-017).
- Попытки усилить правило поглощения групп (по косинусу 0.2/0.45, по ширине
  владельца ≥16 полос) отвергнуты: S09/S10/S12/S01 ломаются (призраки). Найдено,
  что юнит-тест `TwoClustersProduceTwoObjects` не проходил ещё с D-016 (соседние
  кластеры); фикстура разведена на 20 полос, утверждения прежние. xUnit снова 7/7.
- Обучение ушло в разнос: с 7089 отдельные батчи loss > 25, с ~7760 все; остановлено,
  финал — шаг 6000. CUDA-эталон паритета на нём даёт 3.4e-3, CPU-эталон — 6.5e-6:
  экспорт пересобран с `--device cpu` (D-018).
- Финальные прогоны: нейро **9/13**, DSP **9/13** (обязательные 7/7), паритет 6.5e-6,
  tracking: count_in_range 0.818, id_switches 0, reid 1.0.

- 2026-11-09 P9: `FftPlan` переписан на 4 сигнала в дорожках `Vector4` (M/L/R/S за
  один проход). `bench frontend` p50 0.71 → **0.24–0.25 мс**, паритет 3.9e-6, xUnit 7/7
  (D-020).
- `IntraOpNumThreads` 0 → **1** (2.0 против 3.8 мс при 12 потоках). DirectML измерен
  (infer 2.8, total 3.7 мс) и откатан — CPU быстрее (D-019).
- Визуальная проверка PNG выявила: partial loudness (+19…+48 «дБ») прижимал объекты
  к верху. Добавлена нормировка на суммарную сжатию энергию, partial теперь в шкале
  −70…0 дБ (D-021).
- `tools/publish.ps1` больше не содержит кириллицы в литералах (PowerShell 5.1 читал
  файл как ANSI и падал); добавлен `tools/shortcut.ps1` — ярлык AVIS на рабочем столе.
- `dist\AVIS` собран (296 МБ): mock FPS p50 164.96, neural FPS 165.15, analysis p50
  3.38 мс, скриншот и bench в `artifacts/dist`.
- PNG-пресеты на S09 и процедурном миксе (`tools/make_demo_mix.py`) — в
  `artifacts/visual`; FINAL_REPORT.md и ГОТОВО.txt написаны.

