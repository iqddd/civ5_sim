# Работа с RL-запусками

Команды выполняются из корня репозитория. На Windows используйте
`.venv\Scripts\python.exe` вместо `python`.
`CONFIG` — JSON-конфиг, `RUN` — каталог результатов, `CHECKPOINT` — файл модели,
`ROLLOUT` — сохранённый сбор, `DEPLOYMENT` — каталог удалённого запуска,
`OUT` — отдельный каталог вывода. Условия опытов и результаты — в
[ledger](experiment-ledger.md).

## Локальный запуск

```powershell
python tools/run_cp_imitation.py prepare-corpus --config CONFIG --out CORPUS_OUT --game GAME_DIRECTORY --workers 4
python tools/run_cp_imitation.py train --config CONFIG
python tools/run_cp_imitation.py evaluate --config CONFIG
python tools/run_ppo.py run --config CONFIG
python tools/run_ppo.py resume --config CONFIG
```

`prepare-corpus` создаёт корпус из карт и параметров CONFIG. Обучение и оценки
пишут результаты в output конфига. Доступные фазы зависят от протокола;
список показывает `python tools/run_cp_imitation.py --help` или
`python tools/run_ppo.py --help`.

## Анализ сохранённого запуска

```powershell
python tools/ctl_rl.py audit --run RUN --out OUT
python tools/ctl_rl.py compare --before BEFORE --after AFTER --expected-n MATCH_COUNT --out OUT
python tools/ctl_rl.py snapshot --run RUN --out OUT --block-size 5 --last 20
```

| Команда | Вход | Выход |
|---|---|---|
| `audit` | Остановленный RUN | Проверки checkpoints, учёта шагов и полноты оценок |
| `compare` | Два каталога оценок на одинаковых случаях | Разница побед и парный bootstrap по мирам |
| `snapshot` | RUN; `--last` ограничивает число updates | JSON, Markdown и `learning-curves-detailed.png` |

Если `audit` возвращает `incomplete` для работающего запуска, используйте
`snapshot` для промежуточного просмотра. У `compare` число миров, количество
bootstrap-выборок и seed задаются `--clusters`, `--samples`, `--seed`.
В snapshot training win rate объединяется по эпизодам, PPO-метрики усредняются
по updates блока. Эти команды читают сохранённые результаты, не запускают матчи.

Повторить открытую серию с выбранной моделью:

```powershell
python tools/ctl_rl.py play-cases --manifest SHOWCASE_MANIFEST --checkpoint CHECKPOINT --out MATCHES_OUT
python tools/view_combat.py MATCHES_OUT --port 8767
```

На входе manifest серии `civ5rl.showcase` и пустой MATCHES_OUT. На выходе —
матчи на сохранённых расстановках и seeds, доступные в локальной веб-панели.

### Замена команды в policy–CP реплее

```powershell
python tools/ctl_rl.py branch-replay --source MATCH_DIRECTORY --command 30 --action move --unit 8 --target 4 3 --check-only
python tools/ctl_rl.py branch-replay --source MATCH_DIRECTORY --command 30 --action move --unit 8 --target 4 3 --checkpoint CHECKPOINT --out BRANCH_OUT
```

Нужны исходный матч с `case.json`, checkpoint его политики и пустой BRANCH_OUT.
Номер команды начинается с единицы, координаты — с нуля. `--check-only`
проверяет законность замены без доигрывания. На выходе — доигранный матч,
replay, result, final и case. Ветка появляется после обновления каталога панели.

| Флаг | Действие |
|---|---|
| `--action skip` / `fortify` | Ожидание/укрепление; цель не нужна |
| `--actions FILE.json` | Последовательность команд вместо одной замены |
| `--forbid-move-to X Y` | Запрет move/swap на клетку до конца текущего хода |
| `--continue-to-round N` | Продолжить сохранённый бой до указанного раунда |
| `--device cpu` | Выбрать CPU вместо CUDA |

Файл `--actions` содержит непустой массив, например
`[{"kind":"move","unit":8,"target":39}]`, где target — `y*width+x`.
Последовательность начинается с `--command`, затем играет модель.

## Прогресс процесса, EMA и ETA

```powershell
python tools/ctl_rl.py progress --run RUN
python tools/ctl_rl.py progress --run RUN --watch --interval 900
python tools/ctl_rl.py progress --deployment DEPLOYMENT --watch --interval 900 --json
python tools/ctl_rl.py progress --run RUN --alpha 0.1 --exclude-compilation
```

Без `--watch` выводится один срез. `--interval` задаётся в секундах;
`Ctrl+C` завершает просмотр, обучение продолжается. `--deployment` читает
прогресс по SSH, `--run` — с локального диска. После выгрузки и удаления
инстанса используйте `--run`.

| Показатель | Как читать |
|---|---|
| EMA | Сглаженная скорость; `--alpha` задаёт вес последнего измерения |
| PPO decisions/s | Завершённые решения / активное время сбора и PPO |
| BC labels/s или steps/s | Скорость обучения в единицах соответствующего лимита |
| Work ETA | Остаток основной работы при наблюдаемой скорости |
| Other / conditional | Обязательные оставшиеся фазы / условные оценки |
| Active budget | Остаток времени фаз, расходующих активный бюджет |
| `fits` | Помещается ли прогноз в бюджет; неизвестно при недостатке измерений |

EMA обновляется по завершённой работе, не по частоте опроса. После resume
скорость измеряется заново. Текущий незавершённый rollout показывается отдельно
как collected. Development/Additional приостанавливают активные часы:
их время входит в общий ETA, но не в active-budget `fits`.

`--exclude-compilation` вычитает записанное время Dynamo callbacks. Это оценка
без измеренной компиляции, а не ускорение работающего процесса. До первых
измерений скорость и ETA неизвестны. Для paused/error ETA означает время
работы после возобновления; простои и будущие сбои в него не входят.

Автоматический срез находится в `RUN/derived/progress/latest.json` и может
отставать от процесса примерно на интервал записи. Постоянной выгрузки всего
RUN в фоне нет. Ошибка SSH при просмотре не останавливает обучение.

## Пауза и закрытие

```powershell
python tools/ctl_rl.py request-pause --run RUN --after-update UPDATE
python tools/ctl_rl.py request-pause --run RUN --after-epoch EPOCH
python tools/ctl_rl.py wait-and-finalize --run RUN --out OUT --timeout 600
python tools/ctl_rl.py close --run RUN --reason "Research stopped"
```

`request-pause` запрашивает остановку на границе update/эпохи. Дождитесь `paused`;
оценка на этой границе заканчивается до паузы. `wait-and-finalize` ждёт до
`--timeout` секунд и сохраняет промежуточный срез в OUT. `close` закрывает
остановленный запуск с указанной причиной.

Для продолжения используйте `run_ppo.py resume --config CONFIG` либо
`run_cp_imitation.py resume --config CONFIG`, если протокол поддерживает resume.

Продлить активный бюджет:

```powershell
python tools/ctl_rl.py extend-budget --run RUN --additional-seconds 3600 --reason "Extra active hour" --request-id extra-hour-1
python tools/remote_rl.py extend-budget --deployment DEPLOYMENT --additional-seconds 3600 --reason "Extra active hour" --request-id extra-hour-1
```

Поддерживаются after-target-BC PPO и target-scale BC, включая more-data/C2.
`--request-id` позволяет безопасно повторить запрос после обрыва связи.
Status/progress различают записанную и применённую ревизии бюджета.
Прибавка действует на ближайшей безопасной проверке; завершённый запуск
не возобновляется. Лимиты решений и эпох остаются из CONFIG.

## Сравнительное обучение по доигрываниям

Это отдельный экспериментальный runner. Выберите CONFIG по
[ledger](experiment-ledger.md); условия определены конфигом и протоколом.

```powershell
python tools/run_comparison.py precheck --config CONFIG --out PRECHECK_OUT
python tools/run_comparison.py prepare --config CONFIG
python tools/run_comparison.py collect --config CONFIG [--runtime-config RECOMMENDED] [--workers W]
python tools/run_comparison.py train --config CONFIG
python tools/run_comparison.py evaluate --config CONFIG
python tools/run_comparison.py resume --config CONFIG
```

`precheck` использует отдельный PRECHECK_OUT. Остальные фазы пишут в output
конфига; `resume` продолжает незавершённые фазы. Для отдельного сбора
(`rl01.rollout_comparison_collection`) поддерживаются `prepare`, `collect`,
`resume` и опции исполнения (`--runtime-config`, `--workers`, `--source-share`,
`--buffer-limit`, `--pack-mib-per-second`). Состояние и итог — в
`collection.json`, `collection-runtime.json`, `dataset.json`, `training.json`,
`result.json`, `report.md`. Мониторинг — через `ctl_rl.py progress`.

Для обучения на отдельном завершённом сборе выберите CONFIG с `collection_input`
и выполните `train`, затем `evaluate`, либо `resume` для обеих фаз.
Входной корпус проверяется по сохранённым hashes; результаты пишутся в output
обучения. Исходный сбор не повторяется.

### Измерение CPU/GPU сравнительного цикла

```powershell
python tools/bench_rl.py comparison --run RUN --config CONFIG --out OUT --warmup 1 --steps 3
python tools/bench_rl.py comparison-collection --run RUN --config CONFIG --out OUT --worker-counts 6,8,12,16 --forks 64 --seconds 300
```

Вход — сохранённые training-истории (или завершённый сбор для
`comparison-collection`); OUT должен быть пустым и вне RUN. Выход —
`summary.json`, телеметрия CPU/NVML и Kineto-трассы (или `recommended-runtime.json`
и прогноз развилок за бюджет для `comparison-collection`).

## CPU/GPU-инференс оценок

```powershell
python tools/run_ppo.py run --config CONFIG --evaluation-device cuda --evaluation-workers 8 --evaluation-max-batch 32 --evaluation-wait-seconds 0.001
python tools/run_cp_imitation.py evaluate --config CONFIG --evaluation-device cpu
```

| Флаг | Значение |
|---|---|
| `--evaluation-device auto` | CUDA при наличии, иначе CPU |
| `--evaluation-device cpu` / `cuda` | Явный выбор; CUDA без устройства выдаёт ошибку |
| `--evaluation-workers` | Число параллельных матчей симулятора/CP |
| `--evaluation-max-batch` | Максимум запросов в inference batch |
| `--evaluation-wait-seconds` | Предельное ожидание готовых запросов |

Эти флаги доступны для target-scale BC/PPO и after-target-BC PPO, а также
передаются через `remote_rl.py launch`. Если CONFIG закрепляет runtime,
используйте указанные в нём значения. Выбор сохраняется в `evaluation-runtime.json`.
Одновременно batch ограничен числом workers. Оценки используют FP32 без AMP/TF32.
На Windows CUDA-командам нужен активированный MSVC environment (`vcvars64.bat`).
После ошибки повторите оценку с тем же runtime: завершённые матчи сохраняются.

## Производительность и GPU

```powershell
python tools/ctl_rl.py performance --run RUN --last 10 --json
python tools/ctl_rl.py overhead --run RUN --out OUT --reference-start START_UPDATE --reference-stop STOP_UPDATE --through-update UPDATE --deployment DEPLOYMENT
```

`performance` читает законченные updates; без `--last` — весь RUN.
Для записи результата задайте `--output`. `overhead` сохраняет разбор затрат
в OUT; reference range задаёт окно сравнения скорости.

| Показатель | Как читать |
|---|---|
| `collection` | Время сбора |
| `ppo_recorded` | Записанное PPO-время, включая подготовку backend/precheck |
| `ppo_operation` | Полное время вызова PPO update |
| `ppo_operation_without_compilation` | То же без записанных Dynamo callbacks |
| `dynamo_callbacks` | Время tracing/backend callbacks |
| GPU Core Load | Активность GPU по NVML; оценивать вместе с мощностью и trace |
| Memory Controller Load | Активность контроллера памяти, не заполнение VRAM |
| Power / % TDP | Мощность GPU; помогает заметить сериализованную работу |
| Coverage | Доля интервала, покрытая замерами; пропуски не равны нулевой загрузке |

Сравнивайте одинаковые фазы после прогрева. Среднее за весь запуск смешивает
сбор, обучение и оценки. Вложенные KL/backward/cache timers не складывайте
с полным временем PPO. `overhead` показывает также неатрибутированный остаток.

OOM-тренд смотрите в `telemetry/cuda-memory.jsonl` на одинаковых границах updates:

- Растёт allocated — увеличивается объём живых тензоров.
- Растёт reserved при стабильном allocated — увеличивается удержание памяти,
  это само по себе не доказывает утечку.
- Падает driver-free при стабильном allocated — проверьте private pools и другие
  потребители GPU; стабильный allocated не исключает расход CUDA Graphs.

При OOM сохраняется `telemetry/cuda-oom-SESSION-TIME.json.gz`.
Для планового перезапуска запросите паузу до исчерпания VRAM, выгрузите RUN
и выполните resume. Процесс после resume повторяет прогрев/компиляцию.

## Replay, полный цикл и диагностика

```powershell
python tools/bench_rl.py replay --run RUN --compare-git-rev HEAD --out OUT
python tools/bench_rl.py full-cycle --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --out OUT --repeats 3 --device cuda
python tools/bench_rl.py bc --config CONFIG --checkpoint CHECKPOINT --run CORPUS_RUN --scratch-dir .tmp/bc-bench --warmup 3 --steps 5 --trace-steps 3 --verify-parity --parity-steps 3 --out OUT
python tools/bench_rl.py evaluation --run RUN --checkpoint CHECKPOINT --out OUT --cases 2 --workers 4 --warmup 3 --steps 5 --trace-steps 3
python tools/diagnose_cp_replay.py --rollout ROLLOUT --episode EPISODE --trace-from COMMAND --out OUT
```

| Режим | Что измеряет |
|---|---|
| `replay` | Сохранённый rollout; `--compare-git-rev` сравнивает PPO backend выбранного commit |
| `full-cycle` | Сбор и отбрасываемый PPO update; нужен PPO checkpoint с Adam |
| `bc` | Прогретые учебные шаги; `--verify-parity` добавляет сверку с baseline |
| `evaluation` | Матчи и inference выбранного checkpoint на сохранённых случаях |
| `diagnose_cp_replay.py` | Повтор решений CP; `--trace-from` требует trace-enabled native library |

`--warmup` исключается из замера, `--steps` ограничивает число учебных шагов,
`--trace-steps` сохраняет короткую Kineto-трассу и сводку ядер. Исходный RUN
не изменяется. `--compare-git-rev` поддержан только для replay и не переключает
целиком окружение, модель и native-библиотеки на указанный commit.

## Профилирование collection и подбор runtime

```powershell
python tools/profile_rl_runtime.py cpu --rollout ROLLOUT --episodes 8 --repeat 3 --python-profile --out OUT
python tools/profile_rl_runtime.py collection --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --episodes 32 --envs 32 --workers 8 --reply-size 2 --repeat 3 --out OUT
python tools/tune_rl_collection.py --scheduler streaming --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --episodes 80 --budget-seconds 600 --out OUT
```

Вход — CONFIG, CHECKPOINT и сохранённые случаи ROLLOUT. Выход подбора —
`recommended-runtime.json`; передайте его runner через `--runtime-config`.
`--budget-seconds` включает подготовку и прогрев. `preliminary` означает,
что подтверждение рекомендации не завершено.

Сравнивайте `collection_seconds` и decisions/s на одинаковой работе.
Сумма ожиданий workers может превышать wall time. Для native timings соберите
`tools/build_combat_ai.py --profile --dll-only` и задайте profiler флаг
`--native-library`; итоговую скорость измеряйте обычной библиотекой.

## Удалённое исполнение

Нужны CONFIG, JSON подключения с `instance_id`, `host`, `port` и пустой DEPLOYMENT.

### Аренда и CPU-допуск

```powershell
python tools/remote_rl.py rent --profile 5090 --parallel 2 --cpu-score-min 1250
```

Добавьте согласованные фильтры GPU, цены и сети. Из `SUCCESS` сохраните JSON
подключения для `package --instance`. Для CPU-зависимого сбора применяется
`--cpu-score-min`; для BC без сбора он не нужен. Модели CPU и Single Thread — в
[справочнике](../configs/vast-cpu-ratings.json). Неизвестная модель останавливает
отбор до заполнения справочника.

`--sort-by cpu-score` выбирает кандидатов по убыванию CPU-балла, при равенстве —
по возрастанию цены. По умолчанию используется `--sort-by price`.

### Подготовка, tuning и запуск

Нужны чистое рабочее дерево и HEAD в origin. CARRIER — сохранённый rollout
с открытыми случаями для tuning after-target-BC PPO.

```powershell
git push origin HEAD
python tools/remote_rl.py package --config CONFIG --out DEPLOYMENT --instance INSTANCE_JSON --extra-input CARRIER
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode tune
python tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode run --evaluation-device cuda --evaluation-workers 8 --evaluation-max-batch 32 --evaluation-wait-seconds 0.001
python tools/ctl_rl.py progress --deployment DEPLOYMENT --watch --interval 900
```

`package` собирает входы и описание запуска в DEPLOYMENT. `launch` готовит
окружение и запускает процесс; при поддержке протокола `tune` подбирает runtime.
Результаты хранятся на сервере до `pull`.

Для отдельного сравнительного сбора при `package` задайте
`--tuning-source SAVED_COLLECTION`: каталог завершённого сравнительного опыта.
`launch --mode tune` измерит число воркеров; после `pull --keep-instance`
команда `launch --mode run` использует полученный `recommended-runtime.json`.

### Восстановление и завершение

```powershell
python tools/remote_rl.py status --deployment DEPLOYMENT
python tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode resume
```

После завершения или ошибки сначала выгрузите состояние, затем выполняйте resume.
Повтор `launch` активного job присоединяется к нему. Для переноса paused PPO на
другой Linux-хост подготовьте отдельный deployment того же CONFIG, повторите tune:

```powershell
python tools/remote_rl.py launch --deployment NEW_DEPLOYMENT --mode resume --host-resume-reason "Transfer to another host"
```

После финальных оценок выгрузить, проверить и удалить привязанный инстанс:

```powershell
python tools/remote_rl.py pull --deployment DEPLOYMENT --delete-instance
```

Команде нужны credentials/keyring менеджера Vast. При ошибке доставки или аудита
инстанс сохраняется. Для раздельного выполнения используйте `--keep-instance`,
проверьте delivery receipt и audit, затем команды менеджера:

```powershell
python C:\Users\Roman\.codex\skills\vastai-instance-manager\scripts\vastai_instance.py delete INSTANCE_ID
python C:\Users\Roman\.codex\skills\vastai-instance-manager\scripts\vastai_instance.py inspect INSTANCE_ID
```

### Разделение зон (`repositories`, `inputs`, `mirror`, `scratch`)

| Каталог | Назначение |
|---|---|
| DEPLOYMENT | Подключение, пакет запуска, job и подтверждения доставки |
| RUN | Научные результаты, checkpoints и метрики |
| `CIV5RL_SCRATCH_DIR` | Временные подготовленные входы; не выгружаются с результатами |

### Передача больших наборов JSON

```powershell
python tools/ctl_rl.py archive-evaluations --run RUN
python tools/ctl_rl.py restore-evaluations --run RUN
```

`archive-evaluations` проверяет архив и заменяет `evaluation/` на
`evaluation.storage/`. `restore-evaluations` возвращает распакованные файлы для
просмотра или resume. Формат — PPMd, словарь 256 МиБ, order 32, solid 4 ГиБ.
Финальный `pull` после успешного аудита завершённого запуска архивирует локальные
оценки автоматически. Для paused запуска используйте явную команду;
работающий процесс сначала остановите.

`package`, `launch`, `pull` автоматически упаковывают большие JSON-наборы и
проверяют доставку. На обеих машинах нужен `7zz`, `7z` или `7za`.
При сбое повторите `pull`: подтверждение доставки и аудит находятся в DEPLOYMENT.

### Жизненный цикл deployment

`status` показывает состояние удалённого процесса. `pull` ждёт завершения job,
скачивает результаты в RUN и запускает аудит. По умолчанию инстанс сохраняется;
`--delete-instance` удаляет его после успешной выгрузки и проверки.
