# План P0–P9

Отметка `[x]` ставится только после фактической проверки. Этапы: реализовать → проверить → коммит → тег.

## P0 Основа
- [ ] check_env.ps1 (Windows, GPU/VRAM, диски, git, dotnet, uv, curl, tar)
- [ ] Установить недостающее, выбрать диск данных (`E:\avis_data`), задать `AVIS_DATA`
- [ ] Структура репозитория, `.gitignore`, docs (SPEC/PLAN/STATUS/AGENTS)
- [ ] tools/bg.ps1, tools/verify.ps1, tools/config.local.json
- [ ] uv venv + PyTorch CUDA, `torch.cuda.is_available()`
- [ ] git init, первый коммит, тег p0
- [ ] Фоновая загрузка данных (fetch_data.ps1 через bg.ps1)

## P1 Python-фронтенд и сценарии
- [ ] frontend.py: веса FE-v1, numpy-стриминг, torch-батч; weights.bin + frontend.json
- [ ] Тесты: numpy==torch; синус 1 кГц ≈ −6 дБ ±1; шум без ступенек >1.5 дБ
- [ ] scenarios.py S01–S13 + expect.json, golden-векторы
- [ ] Тег p1

## P2 Данные и миксер
- [ ] Процедурный синтез + метки; индекс
- [ ] pYIN-метки в фоне (если есть MUSDB)
- [ ] Миксер, метки кадр×полоса, mix.preview 4 PNG
- [ ] Тесты форм/NaN/детерминизма
- [ ] Тег p2

## P3 Модель и обучение
- [ ] M-v1, лоссы, тесты step==forward, L_dc≈0
- [ ] train.py (resume/CSV/чекпоинты), eval.py
- [ ] smoke.yaml → loss падает ≥30%, resume работает
- [ ] v1 запущен в фоне, PID/лог/команда в STATUS
- [ ] Тег p3

## P4 C#-основа
- [ ] Solution, проекты по границам, build.ps1, xUnit
- [ ] C#-фронтенд + паритет golden ≤2e-3, bench ≤0.5 мс
- [ ] AnalysisFrame + JSONL C#/Python
- [ ] IObjectSource, pipeline, Mock, Replay, CLI analyze/render/check/bench/parity
- [ ] Тег p4

## P5 Рендер и UI
- [ ] Offscreen D3D11 + bloom + PNG-энкодер, тест пикселей, bench ≤3 мс
- [ ] WinUI 3: SwapChainPanel, Mica, CommandBar, статус-строка
- [ ] WASAPI loopback + ресемплер (<−40 дБ искажений)
- [ ] AppSettings, пресеты, автопанель
- [ ] --bench/--autoclose/--set/--screenshot, FPS ≈ монитору
- [ ] Тег p5

## P6 ObjectParams и DSP
- [ ] ObjectParams
- [ ] avis-cli check все типы
- [ ] DspObjectSource: S01–S05, S07, S08 PASS
- [ ] Тег p6

## P7 Нейро-рантайм
- [ ] export.py: step, ONNX opset17, паритет ≤1e-3, color_pca, model.json
- [ ] NeuralObjectSource + C# паритет ≤1e-3, bench ≤1.5 мс
- [ ] ObjectTracker + юнит-тесты (2 кластера, re-ID, слияние, шум)
- [ ] Тег p7

## P8 Интеграция и качество
- [ ] v1 экспортирован и подключён в App/CLI
- [ ] verify -Source neural S01–S13 + eval_tracking.py, таблица в TRAINING.md
- [ ] Задержка p95 ≤30 мс; CPU vs DirectML замер
- [ ] Цикл улучшения ≤5 итераций, таблица гипотез
- [ ] Тег p8

## P9 Полировка и сдача
- [ ] Аккорды а/б, шлейфы, водопад, вспышки, цвет по тембру, оверлей, пресеты
- [ ] README, USER_GUIDE, ARCHITECTURE, TRAINING, TROUBLESHOOTING (рус.)
- [ ] publish.ps1 → dist\AVIS\Avis.App.exe, run.ps1, ярлык
- [ ] Полный verify.ps1
- [ ] FINAL_REPORT.md + ГОТОВО.txt, всё закоммичено
- [ ] Тег p9
