# План P0–P9

Отметка `[x]` — фактически проверено. Этапы: реализовать → проверить → коммит → тег.

## P0 Основа
- [x] check_env.ps1 (Windows, GPU/VRAM, диски, git, dotnet, uv, curl, tar)
- [x] Установить недостающее, выбрать диск данных (`E:\avis_data`), задать `AVIS_DATA`
- [x] Структура репозитория, `.gitignore`, docs (SPEC/PLAN/STATUS/AGENTS/DECISIONS)
- [x] tools/bg.ps1, tools/config.local.json (verify.ps1 — см. P4/P9, готов)
- [x] uv venv + PyTorch CUDA, `torch.cuda.is_available()`
- [x] git init, первый коммит, тег p0
- [x] Данные: процедурный миксер (D-001); fetch_data.ps1 оставлен для опциональной загрузки

## P1 Python-фронтенд и сценарии
- [x] frontend.py: веса FE-v1, numpy-стриминг, torch-батч; weights.bin + frontend.json
- [x] Тесты: numpy==torch; синус 1 кГц ≈ −6 дБ ±1; шум без ступенек >1.5 дБ
- [x] scenarios.py S01–S13 + expect.json, golden-векторы
- [x] Тег p1

## P2 Данные и миксер
- [x] Процедурный синтез + метки; индекс
- [x] pYIN-метки не требуются (нет MUSDB); процедурные метки питча
- [x] Миксер, метки кадр×полоса, mix.preview 4 PNG
- [x] Тесты форм/NaN/детерминизма
- [x] Тег p2

## P3 Модель и обучение
- [x] M-v1, лоссы, тесты step==forward, L_dc≈0
- [x] train.py (resume/CSV/чекпоинты), eval.py
- [x] smoke.yaml → loss падает ≥30%, resume работает (val 1.4154)
- [ ] v1 обучен 12000 шагов и экспортирован (идёт, PID 17476)
- [ ] Тег p3 (после v1)

## P4 C#-основа
- [x] Solution, проекты по границам, build.ps1, xUnit
- [x] C#-фронтенд + паритет golden ≤2e-3 (факт 3.9e-6), bench ≤0.5 мс
- [x] AnalysisFrame + JSONL C#/Python
- [x] IObjectSource, pipeline, Mock, Replay, CLI analyze/render/check/bench/parity
- [x] Тег p4

## P5 Рендер и UI
- [x] Offscreen D3D11 + bloom + PNG-энкодер, тест пикселей, bench ≤3 мс
- [x] WinUI 3: SwapChainPanel, Mica, CommandBar, статус-строка
- [x] WASAPI loopback + ресемплер; `--wav` для детерминированных тестов
- [x] AppSettings, пресеты, автопанель
- [x] --bench/--autoclose/--set/--screenshot (+ --screenshot-at); FPS 60 (D-014)
- [x] Тег p5

## P6 ObjectParams и DSP
- [x] ObjectParams
- [x] avis-cli check все типы
- [x] DspObjectSource: S01–S05, S07, S08 PASS (9/13 всего)
- [x] Тег p6

## P7 Нейро-рантайм
- [x] export.py: step, ONNX opset17, паритет ≤1e-3, color_pca, model.json
- [x] NeuralObjectSource + C# паритет ≤1e-3; bench p50 ≤1.5 мс
- [x] ObjectTracker + юнит-тесты (2 кластера, re-ID, слияние, шум)
- [ ] p99 total ≤4 мс (4.0–4.7 при фоновом обучении) — перемерить
- [ ] Тег p7 (после перезамера)

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
