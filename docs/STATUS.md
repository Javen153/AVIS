# STATUS (обновляется после каждого шага)

Текущий этап: **P8 — интеграция и качество** (обучение v1 и нейро-проверка)
Обновлено: 2026-11-09 04:30

## Работает
- Окружение: Windows x64, RTX 3050 Laptop 4096 MiB, .NET SDK 10.0.403, git, uv, torch CUDA.
- FE-v1: numpy-стриминг = torch-батч, `shared/frontend/v1/weights.bin` + `frontend.json`.
- Сценарии S01–S13: WAV + expect.json + golden; `tools/scenarios.ps1`.
- Данные: процедурный миксер (Slakh 104 ГБ не скачивался, см. DECISIONS).
- Модель M-v1, обучение smoke (val 1.415) и v1 (в фоне, см. ниже).
- C#: Core/Audio/Render/Cli/App собираются x64 Release; xUnit 7/7.
- Паритет: frontend 3.9e-6; smoke-модель 1.7e-6 (лимит 1e-3).
- DSP: 9/13 (обязательные S01–S05, S07, S08 — PASS).
- Рендер offscreen: PNG-тест PASS, submit p50 0.7–1.2 мс.
- App: mock/DSP/neural через `--wav`, скриншоты, bench, оверлей, настройки.
- Инструменты: build.ps1 (-Test), verify.ps1, scenarios.ps1, publish.ps1, run.ps1, bg.ps1.

## Сломано / неизвестно
- Нейро-сценарии S01–S13 не проверены: ждёт `models/v1/model.onnx`.
- `bench model`: p50 1.0–1.2 мс (PASS), p99 total 4.0–4.7 мс при работающем обучении (лимит 4.0) — перемерить после обучения.
- App FPS p50 60 (таймер UI), цель — частота монитора; зафиксировать в отчёте.
- Нейро-цвет по тембру (PCA) включён, но визуально не оценён на v1.

## Фоновые процессы
- v1 training: PID 17476, `logs/bg/v1.log`, launcher `tools/run_v1_train.cmd`,
  12000 шагов, ~1.5 steps/s (~2 ч). На 04:30 — шаг ~3200.

## Следующие шаги
1. Дождаться v1 → `avis_ml.export` → `models/v1`, parity, scenarios neural, bench.
2. Довести p99 ≤4 мс и App FPS (таймер/презентация).
3. Полный `tools/verify.ps1` → report.md.
4. publish.ps1 → `dist\AVIS`, запуск из dist.
5. README/USER_GUIDE/ARCHITECTURE/TRAINING/TROUBLESHOOTING, FINAL_REPORT, ГОТОВО.txt.
