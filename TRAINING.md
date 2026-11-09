# Обучение M-v1

## Данные

Реальные датасеты (Slakh2100 ~104 ГБ, MUSDB18-HQ, MoisesDB) не скачивались:
на диске `E:` 369 ГБ, но загрузка/разбор не вписывались в бюджет проекта.
Обучение идёт на **процедурном миксере** (`avis_ml/mixer.py`, `synth.py`):

- источники: аддитивные пилы/синусы с вибрато, формантный «голос», барабаны
  (kick/snare/hihat), шум (white/pink/brown), аккорды;
- до 8 источников, панорама, громкость, лёгкая реверберация;
- метки на кадр×полосу: доминирующий источник, активность, питч-гауссово
  распределение, салиентность.

Проверка форм/NaN/детерминизма — `ml/tests/test_model.py`.

## Модель

`AvisNet`: вход `feat [B,T,3,192]`, состояние GRU `[2,B,384]`.
2 слоя GRU × 384; головы:

- `emb` — 20-мерный L2-эмбеддинг каждой полосы;
- `act` — sigmoid активности;
- `pitch` — sigmoid-профиль по 192 полосам;
- `sal` — салиентность.

`forward` (обучение) и `step` (инференс) сверяются тестом.

## Лоссы

```
L = L_dc + 0.5*L_act + 1.0*L_pitch + 0.5*L_sal
L_dc   — deep clustering (эффективная форма)
L_act  — BCE
L_pitch— BCE с весом положительных 10
L_sal  — MSE по активным полосам
```

AMP включён; лоссы считаются вне autocast. Оптимизатор AdamW, warmup 1000,
lr 1e-3, batch 8, T=64.

## Запуск

```powershell
cd ml
$env:PYTHONPATH = (Get-Location).Path
uv run python -m avis_ml.train --config configs/smoke.yaml
uv run python -m avis_ml.train --config configs/v1.yaml --resume
uv run python -m avis_ml.eval  --ckpt models/v1/ckpt_best.pt --batches 4
uv run python -m avis_ml.export --ckpt models/v1/ckpt_best.pt --out ..\models\v1 --device cpu
```

Фоновый запуск: `tools\bg.ps1 -Name v1 -Cmd "tools\run_v1_train.cmd"`.
Лог: `logs/bg/v1.log`, PID: `logs/bg/v1.pid`.

## Результаты

| Модель | Шаги | val loss | L_dc | L_act | L_pitch | L_sal |
|---|---|---|---|---|---|---|
| smoke | 500 | 1.4154 | 0.1961 | 0.0780 | 1.1530 | 0.0546 |
| v1 (промежуточный, ~3k) | 3000 | 1.2670 | 0.1808 | 0.0441 | 1.0374 | 0.0534 |
| v1 (финал) | 12000 | _обновить_ | | | | |

Eval на процедурной валидации (промежуточный чекпоинт):

| Метрика | Значение |
|---|---|
| cell_acc (oracle-K, Hungarian) | 0.548 |
| pitch precision / recall / F1 | 0.844 / 0.629 / 0.720 |
| act IoU | 0.986 |

`eval_tracking.py` (прокси по JSONL: count_err, доля кадров в диапазоне,
смены ID на секунду, re-ID после паузы ≥0.4 с) — таблица обновляется после
экспорта финальной модели; результаты в `artifacts/eval/tracking.json`.

## Железо и время

- RTX 3050 Laptop 4 ГБ, batch 8, AMP: ~0.6–1.2 с/шаг при параллельной работе;
  12000 шагов ≈ 2 ч.
- Экспорт ONNX (CPU) ≈ 10 с; паритет PyTorch/ONNX Runtime ≈ 1.6e-6.
- Инференс ONNX CPU: p50 ≈ 1.0–1.2 мс/кадр (цель ≤1.5).

## Ограничения

- Домен реальных записей не покрыт; возможны ошибки на живых инструментах.
- Питч-голова уверенно работает в среднем регистре; крайние частоты хуже.
- 20-мерный эмбеддинг ограничивает разделение похожих тембров.
