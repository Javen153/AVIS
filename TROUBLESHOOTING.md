# Решение проблем

## Нет звука / пустая сцена
- WASAPI loopback в тишине не отдаёт данных. Включите воспроизведение.
- Проверьте устройство вывода Windows по умолчанию — AVIS захватывает именно его.
- Если приложение поймало «нет устройства», оно переключится на Mock (видны
  синтетические объекты). Вернитесь на Neural/DSP после включения звука.
- Для проверки без звука: `Avis.App.exe --wav test.wav --source dsp`.

## «Нейро недоступен» / приложение в DSP
- Нет `models\v1\model.onnx`. Соберите/скопируйте модель:
  ```powershell
  ml\.venv\Scripts\python.exe -m avis_ml.export --ckpt ml\models\v1\ckpt_best.pt --out models\v1 --device cpu
  ```
- Или укажите другой каталог: `--model models\smoke` (путь или имя).
- ONNX Runtime требует x64; при сборке AnyCPU возможны ошибки загрузки.

## Низкий FPS
- На этой машине (Intel UHD, панель 165 Гц, 1422×730) mock даёт p50 ≈165 FPS:
  рендер идёт в отдельном потоке с vsync `Present(1,0)`.
- Если FPS просел, проверьте `--source mock --bench 5`: mock высокий → узкое место
  в модели/анализе; низкий → окно, драйвер или композитор.
- Уменьшите размер окна или `visual.bloom`, `visual.splatSize`.
- CPU-инференс: `bench model`. DirectML можно включить в настройках (model.provider),
  но на 4 ГБ VRAM возможны компромиссы.

## Артефакты/белый экран в снимке
- PNG пишется премультиплицированным; AVIS принудительно ставит alpha=255.
- Если снимок пуст, проверьте, что звук реально идёт и `objects_avg > 0`
  в `bench.json`.

## Оверлей не появляется
- `F11` или кнопка «Оверлей»; `Esc` — выход.
- В эксклюзивном fullscreen некоторых игр оверлей не виден — используйте
  borderless/windowed.
- Оверлей «сквозной» для мыши: вернуть управление можно только `F11`/`Esc`
  (горячие клавиши работают, пока окно активно).

## Сборка
- `MSB4126: недопустимая конфигурация Release|x64` — используйте `tools\build.ps1`
  (он собирает проекты по отдельности с `-p:Platform=x64`).
- Windows App SDK 1.7: отсутствуют задачи AppxPackage. Проект зафиксирован на 2.6.5.
- Ошибка `Microsoft.Build.Packaging.Pri.Tasks.dll` — та же причина, обновите/понизьте
  WindowsAppSDK до 2.6.5.
- Долгая первая сборка — восстанавливаются NuGet-пакеты (ONNX Runtime, Vortice).

## Проверки и тесты
- `tools\verify.ps1` пишет отчёт в `artifacts\verify\<дата>\report.md`.
- `frontend` FAIL — проверьте `shared\frontend\v1\weights.bin` и golden.
- `model` SKIP — нет `models\v1\model.onnx`; это не ошибка сборки.
- `scenarios` FAIL с «count ...» — модель/трекер не разделяют источники; смотрите
  `docs/DECISIONS.md` и `FINAL_REPORT.md`.
- `render` FAIL «0 lit samples» — выбран пустой кадр; проверьте `--every`.

## Обучение
- OOM на 4 ГБ: уменьшите `batch_size` до 4 или `seconds` до 2–3.
- Медленно: закройте браузер/OneDrive; GPU занят другим процессом.
- Прерванное обучение: `--resume` подхватит `ckpt_last.pt`.
- `torch.cuda.is_available() == False` — переустановите torch CUDA в `ml/.venv`.

## Данные
- `fetch_data.ps1` не запускался: обучение на процедурном синтезе. Для реальных
  данных задайте `AVIS_DATA` (минимум 120 ГБ свободно) и запустите скрипт в фоне.
