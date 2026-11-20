# AVIS — визуализатор звука, который «слышит объекты»

AVIS захватывает звук с вывода Windows (WASAPI loopback), выделяет в нём отдельные
источники — ноты, голоса, барабаны, шум — и рисует их как объекты в реальном времени.
Каждый объект живёт во времени: рождается, движется по частоте и панораме, меняет
яркость от громкости и цвет от тембра, а при пропадании плавно гаснет.

## Скачать

Готовая сборка для Windows 10 19041+ / Windows 11 (x64) — на странице релиза:
**https://github.com/Javen153/AVIS/releases/latest**

1. Скачайте `AVIS-1.0.0-win-x64.zip` и распакуйте целиком.
2. Установка с ярлыками: `powershell -ExecutionPolicy Bypass -File install.ps1`
   (копирует в `%LOCALAPPDATA%\Programs\AVIS`; удаление — тот же скрипт с `-Uninstall`).
   Либо просто запустите `Avis.App.exe` — portable-режим.
3. Требования: Windows x64, ничего доустанавливать не нужно (.NET и Windows App SDK внутри).
   Подробности — `README-INSTALL.txt` в архиве.

## Быстрый старт (из репозитория)

1. Запустите `dist\AVIS\Avis.App.exe` (папку `dist\AVIS` можно переносить целиком).
2. Включите музыку в любом приложении — визуализатор сам подхватит системный звук.
3. Кнопки в верхней панели: источник (Neural / DSP / Mock), пауза, настройки, снимок.
4. `F11` — полноэкранный оверлей, `Esc` — выйти из оверлея.

Подробнее — `USER_GUIDE.md`. Если что-то не работает — `TROUBLESHOOTING.md`.

## Что внутри

- **FE-v1** — причинный слуховой фронтенд: 192 лог-полосы, четыре разрешения
  (4096/2048/1024/512), обновление 100 раз в секунду. Одинаково считается на Python
  (numpy/torch) и C#; паритет ≤2e-3.
- **M-v1** — рекуррентная модель (GRU 2×384): на каждый кадр выдаёт эмбеддинг
  каждой полосы, активность, питч и салиентность. ONNX Runtime, CPU.
- **ObjectTracker** — кластеризация полос по эмбеддингам, рождение/слияние/затухание
  объектов, банк «отпечатков» для повторной идентификации.
- **D3D11-рендер** — аддитивные сплаты, bloom, шлейфы/водопад, HDR → sRGB.
- **WinUI 3** — окно с Mica/Acrylic/чёрным фоном, панель настроек со сбросом,
  подсказки, пресеты, оверлей.

## Требования

- Windows 10 19041+ / Windows 11, x64.
- .NET 10 SDK — только для сборки из исходников (готовый `dist` self-contained).
- Для обучения: Python 3.11 + `uv`, NVIDIA GPU с CUDA (проверено на RTX 3050 4 ГБ).

## Сборка и проверка (для разработки)

```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
tools\check_env.ps1
tools\build.ps1 -Test          # Release x64 + xUnit
tools\verify.ps1               # build/unit/frontend/model/scenarios/render/app
tools\publish.ps1              # dist\AVIS
tools\run.ps1                  # запуск dist или build-выхода
```

Отчёт проверки: `artifacts/verify/<дата>/report.md` и `report.json`.
Лимиты и фактические числа — в `FINAL_REPORT.md`, решения — в `docs/DECISIONS.md`.

## CLI

```powershell
avis-cli analyze --in song.wav --source neural --model models\v1 --out frames.jsonl
avis-cli render  --frames frames.jsonl --out png --view main --every 10
avis-cli check   --frames frames.jsonl --expect testdata\scenarios\S01.expect.json --out r.json
avis-cli bench   --what frontend|model|tracker|render
avis-cli parity  --golden shared\golden --model models\v1
avis-cli capture-test --seconds 5 --out cap.wav
```

## Структура

```
ml/            Python: фронтенд, синтез, модель, обучение, экспорт
runtime/       C#: Core, Audio, Render, Cli, App, tests
shared/        FE-v1 weights, golden-векторы, схемы
models/<v>/    model.onnx, model.json, frontend.json, weights.bin
testdata/      сценарии S01–S13 (WAV + expect.json)
tools/         build, verify, scenarios, publish, run, bg, status
docs/          SPEC, PLAN, STATUS, LOG, DECISIONS, TRAINING, ...
```

## Долгие задачи (обучение)

Запуск в фоне: задача работает без своего консольного окна, а рядом открывается
окно-табло **«AVIS: &lt;имя&gt;»** — шаг, % готовности, скорость, ETA и время
финиша, текущая фаза и последние строки лога. Окно обновляется каждые 5 секунд и
остаётся открытым после окончания задачи.

```powershell
tools\bg.ps1 -Name v3 -Cmd "uv run python -m avis_ml.train --config configs/v3.yaml" -WorkDir "C:\OpenCode\AVIS\ml"
tools\status.ps1                 # то же табло в текущем окне (все задачи)
tools\status.ps1 -Name v3        # подробно про одну задачу
tools\status.ps1 -Name v3 -Watch 5   # живое обновление, Ctrl+C — выход
```

Весь вывод задачи пишется в `logs\bg\<имя>.log` — там полная история шагов.
Окно-табло можно закрыть, задача от этого не останавливается (в отличие от
закрытия окна самой задачи). `-NoMonitor` — без окна, `-Refresh 10` — реже
обновлять, `-Visible` — показать консоль самой задачи (она будет пустой: текст
идёт в лог).

## Ограничения

- Нейро-источник требует обученную модель `models/v1`; без неё приложение
  автоматически переключается на DSP.
- Эталонная модель обучена на процедурном синтезе. Реальные MUSDB18-HQ/Slakh
  подключались в P12: модель на них получилась хуже по всем метрикам, поэтому
  выдаётся v1 (`docs/DECISIONS.md`, D-033; `TRAINING.md`).
- WASAPI loopback в тишине не даёт данных — включите воспроизведение.
- Оверлей работает поверх всех окон, но в некоторых играх с эксклюзивным
  fullscreen может не отображаться.

Лицензии зависимостей: .NET/WinUI (MIT), Vortice (MIT), NAudio (MIT),
ONNX Runtime (MIT), PyTorch (BSD).
