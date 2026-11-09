# STATUS (обновляется после каждого шага)

Текущий этап: **P9 — полировка и сдача** (P8 закрыт, идёт publish + отчёт)
Обновлено: 2026-11-09 07:35

## Работает
- Окружение: Windows x64, RTX 3050 Laptop 4096 MiB, .NET SDK 10.0.403, git, uv, torch CUDA.
- FE-v1: numpy-стриминг = torch-батч, `shared/frontend/v1/weights.bin` + `frontend.json`.
- Сценарии S01–S13: WAV + expect.json + golden; `tools/scenarios.ps1`.
- Данные: процедурный миксер (Slakh 104 ГБ не скачивался, D-001).
- Модель M-v1, **финал — шаг 6000**, val 1.1236 (`ml/models/v1/ckpt_best.pt`).
- Экспорт `models/v1` с CPU-эталоном; паритет C# ONNX **6.5e-6** (лимит 1e-3).
- C#: Core/Audio/Render/Cli/App собираются x64 Release; xUnit **7/7**.
- DSP: **9/13**; обязательные S01–S05, S07, S08 — PASS.
- Нейросеть: **9/13** (S01,S02,S03,S05,S06,S07,S08,S11,S12).
- Трекинг: id_switches 0.000/с, reid_acc 1.000, count_in_range 0.818.
- Рендер offscreen: PNG-тест PASS, submit p50 0.7–1.2 мс.
- App: mock/DSP/neural через `--wav`, скриншоты, bench, оверлей, настройки.
- App FPS: p50 **161–165** на панели 165 Гц (render-поток + `Present(1,0)`), D-014.
- Задержка DSP p95 **18.8–19.1 мс** включая hop (лимит 30), D-010.
- Инструменты: build.ps1 (-Test), verify.ps1, scenarios.ps1, publish.ps1, run.ps1, bg.ps1.

## Сломано / неизвестно
- Нейро S04 (1 объект вместо 2), S09 (ID барабанов), S10 (голос в шуме),
  S13 (слабый saw под шумом) — качество эмбеддингов/питча модели, разбор в FINAL_REPORT.
- `bench model` p99 total 4.0–4.7 мс измерен под обучением; перемерить на простое.
- Нейро-цвет по тембру (PCA) включён, но визуально не оценён на финальной модели.
- Реальных записей нет (нет Slakh/MUSDB): визуальная проверка на процедурных миксах.

## Фоновые процессы
- Нет. Обучение v1 остановлено на шаге 6000 (деградация после 7089, D-018).

## Следующие шаги
1. Полный `tools/verify.ps1` (build/unit/frontend/model/scenarios/render/latency/app).
2. PNG-пресеты (Точный/Красивый/Отладка) на S09 + процедурном миксе, просмотр.
3. publish.ps1 → `dist\AVIS`, запуск из dist с `--bench 5 --autoclose`.
4. FINAL_REPORT.md, ГОТОВО.txt, коммиты и теги p8/p9.
