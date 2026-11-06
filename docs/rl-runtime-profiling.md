# Профилирование Collection на разных CPU

Измеряем фиксированную работу на замороженной политике. Обучение, optimizer,
shuffle RNG и счётчики опыта не изменяются. Меньшая загрузка GPU сама по себе
не является регрессией: основной критерий — время одинакового Collection.

## Воспроизводимый вход

Нужны сохранённый rollout с публичными историями, checkpoint и конфигурация.
Используем один и тот же набор полных эпизодов, seed, CP и веса модели для
всех кандидатов. Последняя волна может содержать меньше эпизодов: Collector
завершает каждый из них полностью. Кратность числу сред или workers не нужна.
Нельзя менять карты, армии, горизонт, поисковые бюджеты CP или reward ради
скорости. Не переносим выбранное число workers между процессорами автоматически.

Команды ниже выполняются из корня проекта через Python окружения опыта.
На Windows здесь используется `.venv\Scripts\python.exe`; на Linux — Python
подготовленного окружения.

## 1. Последовательный CPU replay

```powershell
.venv\Scripts\python.exe tools/profile_rl_runtime.py cpu --rollout artifacts/rl/ppo-after-target-bc/rollouts/latest.pt --episodes 8 --repeat 3 --python-profile --out artifacts/profiling/cpu
```

Один прогрев не входит в медиану. Время включает reset, действия CP, симуляцию
и проверку сохранённых историй. Проверяются точное совпадение публичных
наблюдений, легальных действий, всех команд и результатов. `python.prof`
собирается отдельным дополнительным проходом, вне timed trials.

Для сравнения с прежней реализацией допустим `--source-root` на отдельный
checkout той версии с собранными native-библиотеками и теми же generated rules.
Не переключать исходники и DLL внутри работающего процесса.

## 2. Нативный профиль

```powershell
.venv\Scripts\python.exe tools/build_combat_ai.py --profile --dll-only
.venv\Scripts\python.exe tools/profile_rl_runtime.py cpu --rollout artifacts/rl/ppo-after-target-bc/rollouts/latest.pt --episodes 8 --repeat 1 --warmup 0 --native-library build/combat_ai_profile.dll --out artifacts/profiling/native
```

На Linux библиотека называется `build/libcombat_ai_profile.so`.
Диагностическая библиотека отдельная; production DLL не содержит таймеров.
Счётчики включают geometry build, root import, state export, tactical view,
board refresh, ABI validation и три вида запросов. Времена **inclusive**:
вложенные категории нельзя складывать. Сравниваем число фактически выполненных
операций и доли затрат; итоговое ускорение измеряем обычной библиотекой.

## 3. Полный Collection на CUDA

```powershell
.venv\Scripts\python.exe tools/profile_rl_runtime.py collection --config configs/rl-ppo-after-target-bc.json --checkpoint artifacts/rl/ppo-after-target-bc/checkpoints/update-0006.pt --rollout artifacts/rl/ppo-after-target-bc/rollouts/latest.pt --episodes 32 --workers 8 --reply-size 2 --repeat 3 --out artifacts/profiling/w8-r2
```

Число сред задаётся через `--envs`; без этого флага используется config.
Например, `--envs 12 --workers 12 --reply-size 1` соберёт 32 эпизода волнами
12 + 12 + 8. Workers должны делить ёмкость волны, а `reply-size` — помещаться
в число сред worker. Эти runtime-параметры не переписывают config и случаи.

На каждом новом CPU сначала фиксируем модель CPU, физические ядра, логические
потоки/SMT, доступную RAM, GPU, версии Python/Torch, commit и SHA256 DLL.
Также сохраняются CPU affinity, cpuset, Linux cgroup v1/v2 CPU quota и доступная
память с учётом видимых родительских групп, если доступны:
в контейнере число ядер хоста может превышать выделенную CPU-квоту.
Команда сохраняет эти данные в `result.json`. Не предполагаем, что 12 потоков
равноценны 12 физическим ядрам или что число workers должно равняться одному
из этих чисел.

Кандидаты запускать **последовательно**, без параллельной компиляции, тестов или
других benchmark. Первый полный прогрев исключён. Использовать не менее трёх
timed trials и медиану; при заметном разбросе повторить соседние варианты в
обратном порядке. Фоновая нагрузка и холодная компиляция видны в телеметрии;
не выдавать один выброс за выигрыш реализации.

В `result.json` сравниваем:

- время собственно `collection_seconds` и решения/с;
- inference seconds, количество вызовов и средний batch;
- worker request/wait seconds, publish delay и idle slot seconds;
- точное совпадение command hash; небольшие различия FP32 log-probabilities
  от разных форм inference-batch записаны отдельно, как и recurrent/value;
- `compilation_seconds`, отдельно от установленной производительности.

`idle_slot_seconds` и worker request seconds суммируются по слотам/workers и
могут превышать wall time; это не дополнительные секунды длительности фазы.
Проверочные хеши вычисляются вне показателя `collection_seconds`.

NVML-телеметрия по фазам находится в `resources.jsonl`. Считать GPU Core Load
по непрерывным участкам нужной фазы, исключив warmup и compilation. Не подменять
его WDDM D3D Usage. Правило 80%/60 секунд из AGENTS.md относится к длительным
итеративным GPU-фазам; CPU-bound Collection не следует искусственно удлинять
или загружать бесполезной GPU-работой.

## Выбор и подключение

Выбираем вариант по времени одинакового Collection, сохранив корректность и
фиксированный состав эпизодов. Параметры исполнения передаются в Collector
через `num_envs`/`cpu_workers`/`reply_size`; config, checkpoint и manifest не переписываются.
Для нового runner настройки машины хранятся в отдельном runtime config и
фиксируются в артефактах при следующем явном запуске.

Текущий файл — `configs/rl-ppo-after-target-bc-runtime.json` с ключами
`cpu_workers` и `worker_reply_size`; необязательный `num_envs` задаёт ёмкость волны.
Другой файл передаётся runner через
`--runtime-config <path>`. Это не команда возобновления: запуск обучения
выполняется отдельно по явному решению пользователя.

## Автоматический подбор на Linux

Из корня проекта, с уже собранными native-библиотеками и рабочим CUDA-окружением:
Конфигурация должна использовать `performance.collection_scheduler = "ready"`,
чтобы поддерживать настройку размера ответов.

```bash
python tools/tune_rl_collection.py \
  --config configs/rl-ppo-after-target-bc.json \
  --checkpoint artifacts/rl/ppo-after-target-bc/checkpoints/update-0006.pt \
  --rollout artifacts/rl/ppo-after-target-bc/rollouts/latest.pt \
  --out artifacts/profiling/collection-tune --dry-run
```

Убрать `--dry-run`, чтобы выполнить подбор. `--episodes N` задаёт одинаковое
число случаев для всех вариантов. По умолчанию берутся первые `max(32, 4B)`
эпизодов, но не больше имеющихся, без повторения; `B = max(1, floor(C))`, где
`C` — минимум affinity/cpuset и CPU quota. `--cpu-budget` переопределяет `C`.
Название CPU, число потоков хоста и поле Vast `cpu_cores_effective` полезны для
контекста; реальные ограничения уже запущенного контейнера определяются локально.

Сначала проверяются `W/W/1` около `B/2`, `B`, `1.5B`. При ускорении на верхней
границе не менее 5% добавляется `2B/2B/1`; при победе средней точки — промежуточный
вариант. Затем проверяются две среды на лучший worker и, при выигрыше не менее
5%, три среды. Для лучшего варианта с несколькими средами проверяется reply 2.
Дубликаты и варианты с ёмкостью больше половины выбранных эпизодов пропускаются.

Для явного сравнения, без эвристического расширения:

```bash
python tools/tune_rl_collection.py \
  --config configs/rl-ppo-after-target-bc.json \
  --checkpoint artifacts/rl/ppo-after-target-bc/checkpoints/update-0006.pt \
  --rollout artifacts/rl/ppo-after-target-bc/rollouts/latest.pt \
  --episodes 32 --candidates 12/12/1,16/16/1 \
  --out artifacts/profiling/collection-12-vs-16
```

Каждый запуск изолирован в отдельном процессе: один полный прогрев, затем
измерение. Два лучших кандидата подтверждаются тремя дополнительными замерами
каждый с чередованием порядка. При различии медиан до 3% выбирается меньший
расход workers, затем сред, затем меньший reply. Ожидание готовых результатов
искусственно не добавляется; параметры точности и модели берутся из config.

`--budget-seconds` по умолчанию 900: включает загрузку, компиляцию и прогревы.
По границе времени активный profiler останавливается вместе с дочерними
процессами; очистка ресурсов может занять ещё несколько секунд. Частичные
результаты сохраняются, незавершённое подтверждение помечается `preliminary`.
Ошибки, OOM и несовпадение команд/исходов видны в отчёте; несопоставимый вариант
не участвует в выборе. Варианты с ошибками не заменяются автоматически упрощёнными.

Выход: `summary.json`, таблица в stdout, отдельные каталоги запусков с логами,
телеметрией и численными отклонениями от первого кандидата; готовый
`recommended-runtime.json` для `--runtime-config`. Существующие файлы настроек
и состояние обучения не меняются. При изменении числа сред могут сдвинуться
границы training rollout; это записывается как настройка следующего запуска,
а не маскируется под ускорение прежнего режима.

Перекрытие PPO с подготовкой карт или начальным CP-ходом — отдельная задача.
Перекрытие с полноценным сбором на старой политике, выбор первых завершившихся
эпизодов, меньший поиск CP и сокращённые бои меняют протокол. В эту процедуру
они не входят. Аренда Vast.ai выполняется последующим отдельным этапом.
