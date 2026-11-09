# Работа с RL-запусками

Сверено с CLI и реализацией 9 ноября 2026 года. Команды выполняются из корня
репозитория в окружении опыта. `RUN`, `CONFIG`, `CHECKPOINT`, `ROLLOUT` и
`OUT` ниже — подставляемые пути. Условия обучения, parent и критерии выбора
модели ищите в [журнале](experiment-ledger.md), устройство паспортов — в
[архитектуре](rl-architecture.md#паспорта-исходники-и-resume).

Общий пайплайн RL-01 и его отдельные CLI обучения, управления, оценки,
профилирования и отчётов удалены. Текущие BC/PPO runners исполняют фазы своих
протоколов; они не являются способом возобновить старый пилот с изменённым кодом.
Для генерации корпуса служит отдельная фаза без обучения:

```powershell
python tools/run_cp_imitation.py prepare-corpus --config configs/rl-v1.json --out CORPUS_OUT --game GAME_DIRECTORY --workers 4
```

Она использует `corpus.parent_worlds` и параметры генерации из конфигурации,
не запускает BC/PPO и не создаёт модель. Остальные фазы BC используют уже
подготовленный corpus manifest.

## Анализ сохранённого запуска

```powershell
python tools/ctl_rl.py audit --run RUN --out OUT
python tools/ctl_rl.py compare --before BEFORE --after AFTER --expected-n 128 --out OUT --clusters 16 --samples 10000 --seed 20261103
python tools/ctl_rl.py snapshot --run RUN --out OUT --block-size 5 --last 20
python tools/ctl_rl.py performance --run RUN --last 10
```

`audit` проверяет замороженный паспорт/архив, транзакционный PPO или BC ledger,
CPU checkpoint/Adam/RNG, полноту сохранённых оценок и W/D/L, парное сравнение.
Он не сравнивает текущее рабочее дерево. Для старого паспорта без hash
resolved config выводит `config_hash_verified: false`; недоступные проверки
не объявляются пройденными. Проверка архива не доказывает корректность всех результатов.

Для стабильного аудита удерживается существующий writer lock. Работающий
writer или отсутствие lock дают `incomplete`; для текущего среза есть
`snapshot`. Целостность ledger — свидетельство учёта шагов, а не сходимости.
`compare` читает полные каталоги opponent/mode, проверяет cases, обе стороны,
метаданные и W/D/L, затем делает paired bootstrap по parent-world.
По умолчанию 16 clusters, 10 000 samples, seed 20261103; матчи не запускаются.

`snapshot` создаёт JSON/Markdown и `learning-curves-detailed.png`.
Training win rate объединяется по эпизодам, PPO-метрики усредняются по updates
в блоке; development — сохранённые fixed argmax-vs-CP точки с дедупликацией
одинаковых checkpoints. Он не создаёт новых оценок или учебных шагов.

Для производных файлов используйте отдельный OUT либо `RUN/derived/`,
`RUN/snapshots/`, `RUN/interim-snapshot/`. Исходные checkpoints, RNG,
ledger и результаты не редактируются. Неполный исторический формат не
становится сертифицированным от наличия Markdown-отчёта.

## Пауза и закрытие

```powershell
python tools/ctl_rl.py request-pause --run RUN --after-update 12
python tools/ctl_rl.py request-pause --run RUN --after-epoch 10
python tools/ctl_rl.py wait-and-finalize --run RUN --out OUT --timeout 600
python tools/ctl_rl.py close --run RUN --reason "Research stopped at saved boundary"
```

Запрос — порог в `pause.request`, не сигнал процессу. PPO-after-target-BC
поддерживает update boundary; target-scale и target-more-data imitation —
полностью сохранённую эпоху. Неподдерживаемые сочетания отклоняются. Плановая
оценка на границе заканчивается до паузы; дополнительная final evaluation ради
паузы не запускается. Старый запущенный процесс не приобретает новые hooks от правки файлов.

PPO resume удаляет запрос и продолжает сохранённые model/Adam/RNG/счётчики.
Для `rl01.cp_imitation_target_more_data` фаза `tools/run_cp_imitation.py resume`
безопасно продолжает завершённые волны сбора и состояние обучения (model, Adam,
shuffle RNG, курсор batch и потраченный бюджет) без повтора принятых шагов и оценок;
у ранних BC-протоколов общего pause-resume нет. У target-scale PPO есть также
собственные `pause`/`resume` в `tools/run_ppo.py`. Поддержку нужно выбирать по runner,
а не только по наличию файла запроса.

`wait-and-finalize` ждёт lock/status с конечным timeout и пишет производный
срез: статус paused/technical_error/closed_by_user сохраняется по смыслу.
`close` берёт lock, проверяет ledger и отсутствие неполного update/epoch или
pending evaluation, пишет closure/status. Model/Adam/RNG не меняет;
закрытый запуск отказывает обычному train/resume.

## Производительность и GPU

```powershell
python tools/ctl_rl.py performance --run RUN --last 10 --json
```

Без `--last` анализируется весь запуск. Учитываются только законченные строки
`updates.jsonl`; текущий неполный rollout не включается. Reporter работает без
Torch/GPU и показывает timers, coverage, фазовые NVML means, непрерывные
60-секундные окна и наблюдённые GPU-фазы от десяти минут. Запись файлов
по запросу требует `--output`; обычный вызов читает сохранённую телеметрию.

| Показатель | Смысл |
|---|---|
| `collection` | Wall time сбора, записанный collector |
| `ppo_recorded` | Таймер optimizer после создания backend/precheck |
| `ppo_operation` | Весь `update()`, включая backend и precheck |
| `ppo_operation_without_compilation` | Тот же интервал без пересечения с Dynamo callbacks |
| `dynamo_callbacks` | Tracing/backend callbacks, не исключительно генерация CUDA-кода |

Новые `telemetry/phases.jsonl` и samples имеют session ID и monotonic clock.
Интервалы разделяются на границах фаз; gaps больше `max(5s,3*sample_interval)`,
смена session и сброс часов разрывают покрытие. Missing NVML остаётся missing,
а не нулём. Старые sample-only фазы помечаются `sampled_or_mixed`.
Последняя неполная JSONL-строка живого файла игнорируется; порча внутри — ошибка.

Политика проекта: в каждой устойчивой итеративной GPU-фазе длительностью
не менее десяти минут цель — GPU Core Load от 80%, усреднённый за 60 секунд
по NVIDIA NVML или эквивалентной аппаратной метрике. При меньшем значении
нужно исследовать bottleneck; продолжение требует объяснения по измерениям.
Подготовка, compilation, CPU-only collection/evaluation, паузы и post-processing
не входят в длительность/среднее этой фазы. Нельзя соединять короткие PPO-фазы
через collection или менять модель, precision, batch/budget ради процента.
WDDM D3D Usage не заменяет NVML; короткую фазу не удлиняют ради измерения.
CPU-bound collection оптимизируют по времени одинаковой работы.

`integrate_gpu_interval` принимает секунды от `ResourceMonitor.start()`,
не UTC/абсолютный `perf_counter`. Coverage знаменует весь запрошенный интервал,
включая отфильтрованные фазы. У after-target-BC report также формируются
`performance.json/.md` и `performance-last10.json/.md`.

## Replay, полный цикл и диагностика

```powershell
python tools/bench_rl.py replay --run RUN --compare-git-rev HEAD --out OUT
python tools/bench_rl.py replay --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --device cpu --episode-limit 1 --warmup 1 --repeats 1 --out OUT
python tools/bench_rl.py full-cycle --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --out OUT --repeats 3 --device cuda
python tools/bench_rl.py bc --config CONFIG --checkpoint CHECKPOINT --scratch-dir .tmp/bc-bench --warmup 30 --steps 500 --trace-steps 10 --verify-parity --out OUT
python tools/diagnose_cp_replay.py --rollout ROLLOUT --episode 0 --trace-from 100 --out OUT
```

Replay сравнивает только `engine/ppo_fast.py` выбранного Git commit с рабочим
файлом; модель, native-библиотеки и импортируемые helpers общие текущие.
Это не сравнение двух полных исторических окружений. При несовместимых API
или изменениях нескольких зависимостей нужен отдельный checkout.
Без `--compare-git-rev` сравнительная parity не заявляется.

По умолчанию выполняются прогрев каждой реализации и три ABBA-блока
(12 measured passes). Перед проходом восстанавливаются веса, gradients и
Python/NumPy/Torch/CUDA RNG; optimizer не создаётся, новых матчей нет.
Веса должны остаться побитово прежними, replay diagnostics совпасть при
`rtol=3e-5, atol=2e-7`. T=1-only baseline нельзя использовать для T=0.5 rollout.
Прогрев исключается, compilation внутри measured pass остаётся в wall time.
Выход/телеметрия не перезаписываются. CPU-пример — проверка работы команды,
не измерение представительной GPU-пропускной способности.

`full-cycle` действительно собирает данные и выполняет отбрасываемые updates.
Нужен совместимый PPO boundary checkpoint с PPO Adam; BC Adam не сбрасывается
молча. Входы защищаются hashes, состояния/RNG восстанавливаются между trials.
`bc` прогревает компиляцию вне замеряемого окна, затем измеряет итеративное BC-обучение по NVML (`60s` и интеграл интервала), при `--trace-steps` сохраняет Chrome-трассу Kineto CPU+CUDA и сводку простоев/копирований, а при `--verify-parity` сверяет CE, норму градиента и обновлённые веса с синхронным baseline-циклом, не модифицируя каталог научного прогона.
Для `--compare-git-rev` поддержан только replay. Диагностика CP воспроизводит
сохранённый эпизод; `--trace-from` требует trace-enabled native library.
Она сообщает mismatches и root checks; без этого флага доступна обычная сверка.

## Профилирование collection и подбор runtime

```powershell
python tools/profile_rl_runtime.py cpu --rollout ROLLOUT --episodes 8 --repeat 3 --python-profile --out OUT
python tools/profile_rl_runtime.py collection --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --episodes 32 --envs 32 --workers 8 --reply-size 2 --repeat 3 --out OUT
python tools/tune_rl_collection.py --scheduler streaming --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --episodes 80 --budget-seconds 600 --out OUT
```

Используется один фиксированный набор полных эпизодов, seeds, CP и весов.
Python profiling собирается отдельным проходом. Для native inclusive timers
есть `tools/build_combat_ai.py --profile --dll-only` и profiler
`--native-library build/combat_ai_profile.dll` (Linux: `libcombat_ai_profile.so`).
Вложенные native timers не складываются; итоговую скорость измеряют обычной DLL.

Сравнивайте полное `collection_seconds` и decisions/s, затем command hashes,
численные отклонения, inference batch, compilation, CPU/RAM/NVML.
Сумма worker/slot ожиданий может превышать wall time. Workers не равны
физическим ядрам; сохраняются affinity, CPU quota/cpuset и доступная память.
Кандидаты измеряются последовательно без конкурирующих benchmark.

Автоподбор streaming проверяет ёмкость, workers, reply и объединение inference;
два финалиста подтверждаются чередующимися измерениями. При близости до 3%
предпотение отдаётся меньшему числу workers/сред и меньшему ожиданию/reply.
По умолчанию общий бюджет 600 секунд включает подготовку/прогрев/компиляцию.
Незавершённое подтверждение помечается preliminary; ошибки и несовпадения
не заменяются удобным упрощённым вариантом.

Результат — `recommended-runtime.json`: `num_envs`, `cpu_workers`,
`worker_reply_size`, `collection_scheduler`, `inference_min_batch`,
`inference_wait_seconds`. Он передаётся runner через `--runtime-config`;
подбор не меняет scientific config или paused state и не возобновляет обучение.
Streaming завершает все начатые бои; scheduler/ёмкость могут изменить границы
rollout и фиксируются как условия исполнения.

Подробные scopes `--collection-diagnostics`, gauges/histograms и исторические
замеры сохранены в [прежнем справочнике](archive/operations/rl-runtime-profiling.md).
Диагностическое receive/send время может перекрываться с другой работой;
локальная очередь worker не измеряет задержку runnable-потока планировщиком ОС.

## Удалённое исполнение

`tools/remote_rl.py` работает с уже существующим инстансом поверх `remote-python-ssh`.
Аренда и выбор машины выполняются отдельно; по умолчанию проекта при создании
Vast-инстанса готовятся два кандидата (`--parallel 2`).

```powershell
python tools/remote_rl.py package --config configs/rl-ppo-target-bc-t05.json --out artifacts/deployments/ppo-t05 --extra-input CASE_CARRIER.pt
python tools/remote_rl.py launch --deployment artifacts/deployments/ppo-t05 --mode tune --instance INSTANCE_JSON
python tools/remote_rl.py pull --deployment artifacts/deployments/ppo-t05
python tools/remote_rl.py launch --deployment artifacts/deployments/ppo-t05 --mode run
python tools/remote_rl.py status --deployment artifacts/deployments/ppo-t05
python tools/remote_rl.py pull --deployment artifacts/deployments/ppo-t05 --keep-instance
python tools/remote_rl.py launch --deployment artifacts/deployments/ppo-t05 --mode resume
```

| Kind | run | tune | resume |
|---|---|---|---|
| `rl01.ppo_after_target_bc` | Да | Да | Да |
| `rl01.ppo_target_scale` | Да | Нет | Да |
| `rl01.ppo_scale_up`, включая continuation | Да | Нет | Нет |
| `rl01.cp_imitation_target_more_data` | Да | Нет | Да |
| `rl01.cp_imitation_target_scale` | Да | Нет | Нет |

Каталог `--out` (`DEPLOYMENT`) должен быть пустым и располагаться **вне** `config["output"]`
(например, `artifacts/deployments/<name>`), так как сам `config["output"]` выступает
основным синхронизируемым mirror-каталогом (`--local-sync-dir` <-> `--remote-sync-dir`).
Перед `package` рабочее дерево Git должно быть чистым, а `HEAD` — отправлен в `origin` (`git push`).

### Разделение зон (`repositories`, `inputs`, `mirror`, `scratch`)

1. **Код и зависимости (`prepare.json`)**: удалённая машина клонирует/обновляет публичный
   GitHub-репозиторий на точном `HEAD`-коммите в `<remote_root>/civ5_sim`
   (`execution_subdir: "civ5_sim"`), создаёт `<remote_root>/.venv` и ставит `requirements-rl.txt`.
   На Linux `civ5rl/deploy/worker.py` автоматически собирает `build/libcombat.so` и
   `build/libcombat_ai.so` через `--dll-only`, если библиотеки ещё не собраны.
2. **Входные gitignored-файлы (`DEPLOYMENT/inputs/` -> `prepare.json` `sync`)**:
   `package` отбирает только необходимые входы вне Git (`build/combat_rules.*`,
   `build/cp_geometry.inc`, parent/reference checkpoints, открытые карты манифестов без
   закрытых `test` splits, старый корпус для `cp_imitation_target_more_data` и явные
   `--extra-input`) и раскладывает их по непересекающимся поддеревьям для одноразового
   SHA-256-зеркалирования `prepare.sync`. Они не пересекаются с `config["output"]` и
   не пересылаются обратно при `post-sync`.
3. **Основной вывод опыта (`--local-sync-dir` <-> `--remote-sync-dir`)**: для `run`/`resume`
   синхронизируется непосредственно `config["output"]` (для `tune` — `DEPLOYMENT/tuning`).
   При `pull` (`--attach` + verified `post-sync`) результаты сразу попадают в `artifacts/rl/<run>`
   без промежуточных `result.zip` и дублирующих копий.
4. **Временный кэш (`CIV5RL_SCRATCH_DIR`)**: подготовленные `.npz` файлы BC (`prepared-inputs/`)
   размещаются в `<remote_root>/scratch` вне синхронизируемых каталогов — они сохраняются
   между `resume` на том же хосте и никогда не выгружаются при `post-sync`.

### Жизненный цикл deployment

`INSTANCE_JSON` задаёт `instance_id`/`id`, `host`/`SSH_host`, `port`/`SSH_port`; credentials
берутся из окружения SSH-инструмента. Инстанс после привязки не меняется. Повтор `launch`
при активном job выполняет `--attach`; `resume` требует завершённого `pull` предыдущего
шага (после которого локальный и удалённый `config["output"]` идентичны по SHA-256).

`status` читает ограниченный живой срез по SSH. `pull` дожидается завершения активного job,
проверяет `execution` и `post-sync` без ошибок файлов, запускает научный `audit` напрямую
на `config["output"]` и по умолчанию сохраняет инстанс; `--delete-instance` удаляет
только привязанный ID после прохождения обязательных секций аудита.
