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
- [x] v1 обучен и экспортирован: шаг 6000, val 1.1236 (деградация после 7089, D-018)
- [x] Тег p3 (после v1)

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
- [x] NeuralObjectSource + C# паритет ≤1e-3 (факт 6.1e-6)
- [ ] bench p50 ≤1.5 мс: факт 1.78–1.90 (фон) / 1.74 (P-ядро) — цель не достигнута (D-019, D-022)
- [x] ObjectTracker + юнит-тесты (2 кластера, re-ID, слияние, шум)
- [x] p99 total ≤4 мс: 3.3–3.4 на P-ядре, 4.0–4.5 в фоне
- [x] Тег p7

## P8 Интеграция и качество
- [x] v1 экспортирован и подключён в App/CLI (шаг 6000, паритет 6.5e-6)
- [x] verify -Source neural S01–S13 + eval_tracking.py, таблица в TRAINING.md (9/13)
- [x] Задержка p95 ≤30 мс (DSP 13.1, neural 15.0 мс)
- [x] Цикл улучшения ≤5 итераций, таблица гипотез (D-016, D-017)
- [x] Тег p8

## P9 Полировка и сдача
- [x] Аккорды а/б, шлейфы, водопад, вспышки, цвет по тембру, оверлей, пресеты
- [x] README, USER_GUIDE, ARCHITECTURE, TRAINING, TROUBLESHOOTING (рус.)
- [x] publish.ps1 → dist\AVIS\Avis.App.exe (296 МБ), run.ps1, ярлык (shortcut.ps1)
- [x] Полный verify.ps1: 6 PASS, 2 ожидаемых FAIL (model p50, нейро 9/13)
- [x] FINAL_REPORT.md + ГОТОВО.txt, всё закоммичено
- [x] Тег p9

## P10 Исправление вылета UI (после сдачи)
- [x] Локальное воспроизведение `0xC000027B` при нажатии кнопок и на `--settings`
- [x] Причина 1: стили WinUI (XamlControlsResources) нельзя мержить из кода на
  Windows App SDK 2.6 — добавлен `App.xaml` + `InitializeComponent()` (D-023)
- [x] Причина 2: `dotnet publish` не копировал `Avis.App.pri` — добавлен таргет
  `PublishPriIndex`, PRI теперь в dist
- [x] Титулбар: элемент `SetTitleBar` помещён в дерево (строка 0 корневого Grid)
- [x] Удалена временная диагностика (`Diag.cs`, хуки `AVIS_PANEL_*`, `AVIS_APP_RES`,
  `AVIS_NO_GPU`); оставлен `crash.log` + `UnhandledException`
- [x] `build.ps1 -Test` 7/7, `publish.ps1` (298 МБ), клики по всем кнопкам,
  группам настроек, TextBox и ComboBox на dist — без падений, `--bench 3` код 0

## P11 (после сдачи) — сброс настроек, чёрный фон, подсказки
- [x] `AppSettings.ResetToDefaults()` + кнопка «По умолчанию» в заголовке панели
  (D-024)
- [x] `WindowMaterial.Black`: без системного backdrop, чёрный корневой Grid;
  переключение в живом окне через ComboBox «Материал»
- [x] Подсказки: короткие у кнопок CommandBar и списков, подробные у настроек
  (описание + диапазон/шаг, значение по умолчанию), включая TextBox
- [x] Проверка: реальные наведение/клики (bin и dist) — подсказки видны, сброс
  0,8 → 1, фон Black 94–95 % ↔ Mica 0 %; `build.ps1 -Test` 7/7; `publish.ps1`
  298 МБ, `Avis.App.pri` на месте

## P12 Реальные датасеты и v2
- [x] Читатели проверены на реальных файлах: PyAV (MP4/AAC), soundfile (FLAC, seek),
  pretty_midi; `AVIS_DATA=E:\avis_data` (D-025)
- [x] Индекс `index.jsonl`: 16 558 стемов, 1 161 ч, 599 ч активных, train/test overlap 0
- [x] Словарь категорий и маппинг Slakh/MUSDB (`categories.py`), тесты 13/13
- [x] pyin-кэш MUSDB (12 окон × 4 с, 300/300, 65 мин) (D-026)
- [x] Реальные источники в миксере, измеряемые метки доминанты (D-027)
- [x] Smoke с реальными данными: loss 8.74 → 2.17 за 40 шагов, resume работает
- [x] Замер 0.448 с/шаг, выбрано 16 000 шагов ≈ 2 ч (D-028)
- [x] Прогон 1: остановлен на 3480 из-за коллапса представлений; причина измерена,
  фикс (lr 2.5e-4, clip 1.0, отбраковка тихих чанков, авто-откат) (D-029)
- [x] Детерминизм val: найден недетерминированный `cumsum` на CUDA, val-наборы
  бит-в-бит воспроизводимы, тесты 13/13 (D-030)
- [x] Обучение v2 (16 000 шагов): прогон 4 остановлен на 6000 после третьего отката,
  дообучение от v1 тоже деградировало — v2 хуже v1 (D-031…D-033)
- [x] Экспорт v2 (CPU), паритет PyTorch↔ONNX **1.36e-6** ≤1e-3
- [x] verify S01–S13: v2 **6/13** против v1 **9/13**; tracking на MUSDB/Slakh test,
  cell_acc/pitch F1/act IoU — v2 хуже по всем осям (D-033)
- [x] Визуальная проверка PNG (main + masks) на реальных клипах и S09 —
  `artifacts/eval/vis`, рендер исправен, объектов меньше ожидаемого
- [x] Цикл улучшения: 6 гипотез (lr/clip, детерминизм val, снижение lr, пробы
  proc/real, дообучение от v1), таблица в `TRAINING.md`; `models\v2` не выдаётся

## P13 Разбор Slakh и один повторный прогон
- [x] Диагностика step 1: MIDI↔аудио, 44.1→48 кГц, `is_drum`, 440 Гц/полосы FE-v1,
  доля pitch-кадров и слотов на validation (D-034)
- [x] Причина pitch F1=0: v1 0.374 против v2 0.000 на одних клипах; голова pitch v2
  не выдаёт пиков (норма весов 0.71 против 1.08) — дефект модели
- [x] Исправлен time-warp pitch-меток при speed-аугментации в `mixer_v2`
  (`RealMixer._warp_f0`), тест 440 Гц + warp, `pytest` 15/15 (D-034)
- [x] Один прогон `v2fix` (50/50, якорь v1, 4000 шагов): остановлен на шаге 2000,
  val 2.0116 (+50.9 % к базе 1.3332) — сработал порог отката +10 % (D-034)
- [x] Оценка v2fix: сценарии 5/13, MUSDB 0.404/0.095/0.985, Slakh 0.380/0.000/0.914,
  tracking 1.58/0.428/0.00/1.000 — хуже v1 по критериям; `models\v2` не создан,
  обновлены `TRAINING.md` и `FINAL_REPORT.md`

## P14 v2.1: обучение с нуля по брифу (диагностика, лоссы, декодеры)
- [x] Метрики pitch AP / sweep / IoU / cell_acc, фиксированный val 120+120
- [x] Взвешенный DC (энергия × activity × margin ≥3 дБ), pitch-mask, реальные метки
- [x] Отдельные декодеры на голову, `hid 280` → 2.99 млн параметров
- [x] float32-логиты + BCEWithLogits, guard нефинитных градиентов
- [x] EMA-разогрев; отбор чекпоинта по composite, откат по 2σ шума val
- [x] Сравнение focal / BCE / pos-weight: выбран pos-weight (D-035)
- [x] Smoke и пробы 300 шагов без NaN; ONNX-паритет новой архитектуры 5.4e-7
- [x] Полный прогон v2.1 8000 шагов (`logs/bg/v2_1b.log`, финал AP 0.482,
  composite 1.3900, 204 мин)
- [x] Замер приёмки: сценарии 10/13, Slakh AP 0.514 против 0.400 у v1, MUSDB
  0.110 против 0.130, tracking 0.278/0.904/0.343/0.955, паритет 2.03e-6,
  latency 2.05 мс против 1.49 → **3/6, v2.1 не принята** (D-035)
- [x] Экспорт `models/v2.1`, golden v1 восстановлен
- [x] Выбор модели в настройках (`v1`, `v2.1`), xUnit 9/9

## P15 M-v3: harmonic prior, слим-декодер, стерео, hard cases
- [x] `docs/REQUIREMENTS_V3.md`: цели, архитектура, критерии приёмки
- [x] `model_v3.py`: SHS-prior, depthwise-separable энкодер, GRU 2×208,
  общий декодер + адаптеры; 1.93 млн параметров; контракт шага прежний
- [x] `mixer_v3.py` + `read_chunk_stereo`: стерео-стемы, close_pair,
  pitch_noise, drum_bed; hard_p 0.35
- [x] `arch` в train/validate/export/eval_real; val proc+real+hard
- [x] Независимое ревью M-v3 (субагент): найдены стерео-баг, потеря SNR в
  hard-cases, ложные пики prior на шуме, отбор по hard-val (D-037)
- [x] Исправлены все четыре + `--golden-out`, EMA-экспорт, resume-best,
  `sel_*` в val, форма `h_in` из ONNX-графа; прайор/миксер проверены на CPU
- [x] Preflight 300 шагов после правок: composite 1.163, AP 0.353, IoU 0.956,
  hard 49.8 %, real 12.5 %, NaN нет
- [x] Полный прогон 7000 шагов (`logs/bg/v3.log`): composite 1.3081, AP 0.431,
  IoU 0.995, 114.8 мин; `[mix]` hard 19501, hard_miss 0, стерео 31185
- [x] Сценарии S01–S13 (**7/13**), MUSDB AP 6 клипов 0.1576, Slakh 0.3544,
  tracking 0.000/1.000, latency p50 1.08 мс, ONNX-паритет 2.86e-6
- [x] Экспорт `models/v3`; **не принята** — 7/13, Slakh ниже v1 (D-038);
  эталон остаётся v1, v3 доступна в списке моделей

## P16 M-v3.1: исправление pan-шортката (дообучение v3)
- [x] Диагностика провала: косинус эмбеддингов между активными полосами на
  S04 0.998 (v1 0.59), GRU-абляция не помогает; причина — все hard-пары
  имели противоположную панораму, side-канал был шорткатом (D-038)
- [x] `mixer_v3.py`: 55 % `close_pair` в одной панораме (в т.ч. обе в центре),
  новый hard-case `triad` (2–3 тона в одной панораме), доли семейств
  40/25/22/13; тесты 40/40
- [x] `configs/v3_1.yaml`: дообучение от `ml/models/v3/ckpt_best.pt`,
  1600 шагов, lr 2e-4, hard_p 0.30, реальных ~40 %; preflight 120 шагов — NaN нет
- [x] Прогон 1600 шагов (`logs/bg/v3_1.log`) и оценка `eval_version.ps1`
- [x] Приёмка: **6/13**, реальные AP лучшие (MUSDB 6 кл. 0.1642, Slakh
  0.4305), но ни одного из S04/S09/S13 → не принята; эталон v1 (D-040)

## P17 M-v3.2: solo-семейство (дообучение v3.1)
- [x] Диагностика регрессии S05/S07 по `ownerId`: v3.1 рвёт одиночный
  широкополосный источник на ≈890 Гц; в обучении нет ни одного solo-примера,
  тембры ограничены 2.2 кГц (D-040)
- [x] `mixer_v3.py`: семейство `solo` (пила 1/n до 12 кГц / цветной шум),
  `triad` 25→10 %, доли 35/10/20/20/15; `_labels` для K=1
- [x] `losses.deep_clustering`: fp32-редукция и деление до квадрата
  (устранено переполнение fp16 под AMP); тесты 42/42
- [x] `configs/v3_2.yaml`: от `ml/models/v3_1/ckpt_best.pt`, 900 шагов,
  lr 1e-4; preflight 60 шагов — init 1.3138, solo 23, `hard_miss 0`
- [x] Прогон 900 шагов (`logs/bg/v3_2.log`) и оценка `eval_version.ps1`
  (`ckpt_ema900.pt`, т.к. val не превысил init)
- [x] Приёмка: **8/13** (S05 и S10 восстановлены, S06 сохранён), MUSDB 6 кл.
  0.1602, Slakh 6 кл. 0.4201, `count_in_range 0.727`; ни одного из
  S04/S09/S13 → не принята; эталон v1 (D-041)

## P18 M-v3.3: solo-coherence в лоссе (дообучение v3.2)
- [x] Замер косинуса низ/верх на S05/S07: v3 0.94/0.93, v3.1 0.29/0.33,
  v3.2 0.44/0.49 при пороге трекера 0.60 (`artifacts/tmp/probe_solo.py`)
- [x] `losses.solo_coherence` + флаг `solo` в батче/лейблах, терм вне
  uncertainty weighting; тесты 42/42
- [x] `configs/v3_3.yaml`: от `models/v3_2/ckpt_ema900.pt`, 1200 шагов,
  lr 1e-4, `solo_coh_w 0.5`
- [x] Preflight 100 шагов: `coh` 0.244→0.210→0.209→0.212 (running mean),
  `hard_miss 0`; попутно исправлено логирование `l_coh` (ключ не входил в `lv`)
- [x] Прогон 1200 шагов, оценка, приёмка; итог — D-041 и FINAL_REPORT
- [x] Результат: `coh` 0.244→0.176, cos низ/верх S05 0.564 / S07 0.587
  (порог 0.60), сценарии **8/13** — S12 восстановлен (4/4), S10 регрессировал
  (0/2), S07 всё ещё 2 объекта; реальные AP 0.1612/0.4265 на 6 клипах
- [x] Приёмка: 6/8, **NOT ACCEPTED** (нет ни одного из S04/S09/S13).
  Цикл закрыт: эталон v1, v3.1/v3.2/v3.3 выбираемы


