# Скрипты завершённых опытов

Эти 14 файлов сохранены для чтения и осознанного повторения исторической работы.
Они не являются рекомендуемыми командами следующего этапа. Данные и старые
бинарные файлы могут требовать восстановления из локальных архивов.

| Скрипт | Почему перенесён |
|---|---|
| [benchmark_gru_ablation.py](benchmark_gru_ablation.py) | Завершённое сравнение GRU/MLP/Identity: зафиксированные 512 эпизодов, восемь minibatches, родительский checkpoint и датированный output. |
| [study_cp_reset.py](study_cp_reset.py) | Завершённый CP reset study: конкретный план из 372 матчей и фиксированный корпус. |
| [check_viewer.py](check_viewer.py) | Проверка одной готовой страницы карты: жёсткие 11 карт, 80×52 исходный мир, 16×16 crop и имя download. Для match viewer оставлены общие проверки в корне. |
| [compare_target_selection.py](compare_target_selection.py) | Сравнение завершённого изменения выбора целей с конкретной CP 0.9.3 и историческими каталогами. |
| [measure_tactical_storage.py](measure_tactical_storage.py) | Завершённый перенос ABI5→ABI6 и серия вариантов хранения; преобразование старых запросов относится только к этому опыту. |
| [publish_greedy_replays.py](publish_greedy_replays.py) | Публикация результатов исторической CP/greedy-серии под прежними viewer URL и её архивирование. |
| [summarize_coordinator_acceptance.py](summarize_coordinator_acceptance.py) | Сводка приёмки конкретного этапа CP-07 before/after. |
| [migrate_rl_audit.py](migrate_rl_audit.py) | Узкая миграция checkpoint для уже выполненных audit/pipeline-ревизий; не универсальный resume. |
| [run_rl01.py](run_rl01.py) | Фиксированная последовательность первого пилота C/17: smoke→benchmark→pilot→memory-dev. |
| [run_rl_diagnostic.py](run_rl_diagnostic.py) | Runner завершённых протоколов D0/D1/one-decision; фиксированные критерии и бюджеты. |
| [run_rl_multishot.py](run_rl_multishot.py) | Реализация завершённого D0-M; вызывается диагностическим runner. |
| [report_rl_one.py](report_rl_one.py) | Отчёт только для протокола D0-one-decision. |
| [report_rl_d1.py](report_rl_d1.py) | Отчёт только для ограниченного D1 transfer. |
| [report_rl_multishot.py](report_rl_multishot.py) | Отчёт только для D0-M. |

Для CLI новый путь: `python tools/archive/<script>.py ...` из корня проекта.
Report-модули и `run_rl_multishot.py` — зависимости runner, отдельного CLI у них
нет. Импорты, корень проекта и актуальные ссылки переведены на новое расположение;
обёрток по старым путям нет. Старые snapshots, manifests и политики миграции,
содержащие исторические имена исходников, не переписывались. Строгая проверка
старого опыта после переноса закономерно может отказать из-за изменения
execution identity; обход такой проверки не добавлялся.
