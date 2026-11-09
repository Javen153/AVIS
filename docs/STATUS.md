# STATUS (обновляется после каждого шага)

Текущий этап: **P0 — основа**
Обновлено: 2026-11-09

## Работает
- Окружение проверено: Windows x64, RTX 3050 Laptop 4096 MiB, .NET SDK 10.0.403, git 2.56, uv 0.13.7, curl, tar.
- Диск данных выбран: `E:\avis_data` (E: 369 ГБ свободно).
- Создана структура каталогов и базовые документы.

## Сломано / неизвестно
- Датасеты ещё не скачаны.
- ML-окружение (torch CUDA) не создано.

## Фоновые процессы
- Пока нет.

## Следующие шаги
1. tools/check_env.ps1, bg.ps1, verify.ps1.
2. uv venv + torch CUDA; проверить cuda.
3. git init + commit p0.
4. Запустить fetch_data.ps1 в фоне (Slakh2100 → MUSDB18-HQ → Groove).
5. P1: frontend.py, scenarios.py, golden.
