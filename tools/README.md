# Инструменты проекта

Инвентаризация: 2026-11-05. В корне `tools/` находятся 26 повторно используемых
Python-инструментов и `combat_viewer.html`, ресурс просмотрщика.
14 скриптов завершённых опытов и разовых переносов находятся в
[archive/](archive/README.md).

Рабочий инструмент принимает сценарий, конфигурацию, checkpoint или каталог
результатов и применим к следующему запуску. Скрипт конкретного завершённого
опыта с фиксированными входами, историческими ABI/версиями или заранее заданным
планом запуска хранится в архиве. Наличие `--out` само по себе не делает его
универсальным.

Команды выполняются из корня репозитория:
`.venv\Scripts\python.exe tools/<script>.py --help`.
Сборщик `build_combat_ai.py` запускается без аргументов и не имеет `--help`.

## Сборка и проверка исходников

| Инструмент | Назначение |
|---|---|
| [build_native.py](build_native.py) | Сборка SDK fractal для генерации карт. |
| [build_combat.py](build_combat.py) | Извлечение правил/геометрии CP и сборка ядра боя. |
| [build_combat_ai.py](build_combat_ai.py) | Сборка планировщика CP и нативных проверок. |
| [check_combat_ai_sources.py](check_combat_ai_sources.py) | Проверка checkout CP по зафиксированному списку исходников и SHA256. |

## Матчи, диагностика и просмотр

| Инструмент | Назначение |
|---|---|
| [view_combat.py](view_combat.py) | Локальный просмотрщик произвольного сохранённого матча или набора реплеев. |
| [diagnose_combat.py](diagnose_combat.py) | Разбор выбранных команд/запросов поиска, повтор и альтернативные действия. |
| [resume_combat_replay.py](resume_combat_replay.py) | Продолжение выбранного реплея с сохранением физического состояния и RNG. |
| [tactical_corpus.py](tactical_corpus.py) | Генерация актуальных тактических запросов и проверка исполнения планов. |
| [check_planner_refactor.py](check_planner_refactor.py) | Сравнение любых двух планировщиков текущего ABI на свежих/переданных запросах. |
| [check_combat_viewer.py](check_combat_viewer.py) | Проверка действующего match viewer по переданному URL. |
| [check_combat_refresh.py](check_combat_refresh.py) | Проверка обновления открытого реплея и статусов версий по переданному URL. |

Для обычных матчей также используется `python -m civ5sim match`; отдельный
дублирующий скрипт в `tools/` не нужен. `combat_viewer.html` загружается
`view_combat.py`, поэтому остаётся рядом с ним.

## Подготовка, обучение и оценка RL/BC

| Инструмент | Назначение |
|---|---|
| [prepare_rl.py](prepare_rl.py) | Подготовка корпуса карт по конфигурации. |
| [train_rl.py](train_rl.py) | Обучение/продолжение PPO по переданной конфигурации. |
| [control_rl.py](control_rl.py) | Статус, pause/resume существующего запуска; исторические диагностики делегируются архивному runner. |
| [evaluate_rl.py](evaluate_rl.py) | Оценка checkpoint на разрешённых открытых split. |
| [report_rl.py](report_rl.py) | Отчёт и графики для переданного каталога RL-запуска. |
| [run_cp_imitation.py](run_cp_imitation.py) | Общий CLI фаз prepare/collect/train/evaluate/report и диспетчер настроенных BC-экспериментов. |
| [run_ppo_after_imitation.py](run_ppo_after_imitation.py) | Отдельный ограниченный PPO-опыт от BC: prepare/train/evaluate/report/run, с фиксированным бюджетом и закрытым test. |

`run_cp_imitation.py` оставлен как действующий вход BC: реализации отдельных
опытов находятся в `civ5rl/experiments/`. Перенос старых утилит не ослабляет
проверки происхождения и не разрешает продолжать замороженные эксперименты
после изменения исходников.

## Производительность

| Инструмент | Назначение |
|---|---|
| [benchmark_combat.py](benchmark_combat.py) | Повторные замеры ядра и границы Python/native. |
| [benchmark_combat_ai.py](benchmark_combat_ai.py) | Изолированные матчи CP по выбранному suite, повторы, телеметрия и проверка реплеев. |
| [benchmark_decision_diagnostics.py](benchmark_decision_diagnostics.py) | Накладные расходы off/compact/full на переданных сохранённых запросах. |
| [benchmark_rl.py](benchmark_rl.py) | Повторные полные циклы PPO по конфигурации. |
| [benchmark_rl_pipeline.py](benchmark_rl_pipeline.py) | Сравнение frozen-policy сбора или полного цикла, включая отдельный source root. |
| [benchmark_ppo_replay.py](benchmark_ppo_replay.py) | Замер полного прохода PPO replay/KL с неизменными весами, сравнением кеширования входов и NVML. |
| [profile_rl.py](profile_rl.py) | Сбор фиксированного rollout и CPU-профилей workers для указанной модели. |
| [tune_rl.py](tune_rl.py) | Сравнение реализаций PPO на одинаковом переданном rollout с опциональной трассой. |

Для любого нового замера нужен отдельный каталог результата. Корень `tools/`
не содержит автоматического запуска завершённых исторических опытов.
