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
В target BC `--workers` задаёт число CPU-воркеров для сбора и подготовки
учебных данных. Для оценочных матчей используйте `--evaluation-workers`.

## Анализ сохранённого запуска

```powershell
python tools/run_cp_imitation.py report --config CONFIG --device cpu
python tools/run_ppo.py report --config CONFIG
python tools/ctl_rl.py audit --run RUN --out OUT
python tools/ctl_rl.py compare --before BEFORE --after AFTER --expected-n MATCH_COUNT --out OUT
python tools/ctl_rl.py snapshot --run RUN --out OUT --block-size 5 --last 20
```

Для target BC фаза `report` читает сохранённый запуск и обновляет отчёт и
доступные графики. Для неё достаточно CPU.

| Команда | Вход | Выход |
|---|---|---|
| `run_ppo.py report` | CONFIG и сохранённое состояние RUN | Отчёт опыта по записанным результатам |
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
матчи на сохранённых расстановках и seeds, доступные в локальной веб-панели. Соперника можно
выбрать через `--opponent cp` или `--opponent cpv2`.
По умолчанию действуют правила сброса хода из расстановок. Флаг
`--turn-reset-mode end_turn` или `--turn-reset-mode legacy` задаёт их для всей серии.

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

## Код и сохранённое состояние

Перед обучением или сбором закоммитьте изменения Python/native-кода. Подготовку,
smoke и анализ можно выполнять в рабочем дереве. Первый запуск фиксирует commit
в RUN/execution.json; эта запись хранится вместе с артефактами и не входит в Git.

Для resume прочитайте commit из execution.json и вернитесь к нему в том же
workspace, предварительно сохранив свою работу:

```powershell
git switch --detach COMMIT
python tools/run_ppo.py resume --config RUN/resolved-config.json
```

Для BC и сравнительных сборов используйте их CLI `resume`. Игнорируемые Git
каталоги artifacts остаются на месте при переключении commit. В отдельный
worktree нужно также передать RUN и требуемые входные данные: Git их не копирует.
Сохраняйте весь RUN, включая execution.json и execution-binaries, checkpoints,
исходные корпуса и архивы; без них checkout
кода недостаточен для продолжения. При несовпадении нативных бинарников восстановите их из RUN/execution-binaries
в build. Повторная сборка не обязана давать побитово одинаковые файлы.

После рефакторинга следующий опыт может использовать существующий checkpoint
или завершённый корпус. Переигрывать предыдущие этапы и переписывать их паспорта
не требуется. Для незавершённого запуска без execution.json используйте исходные
инструменты его commit либо начните отдельный опыт от сохранённого checkpoint.

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

Для `training.mode = epoch_kl` (режим по умолчанию) `training.max_epochs`
задаёт число эпох (2), `training.pairs_per_step` — число пар за шаг (32),
`training.initial_lr` — начальный LR (0,00001), `training.kl_limit` — предел
полного KL на границе эпохи (0,015). `training_seconds` ограничивает время
обучения, `workers` задаёт число воркеров сбора и оценки.

Профиль `training.mode = full_corpus_step_kl` показывает принятые шаги,
общее число попыток, отклонения, LR и последний принятый полный KL.
История проверок — в `attempts.jsonl`, принятых шагов — в `steps.jsonl`.
После остановки по KL кандидат содержит последнее принятое состояние.
При `paired_development_only` выполняется Development обеих моделей,
без автоматического Additional. Возобновление восстанавливает незавершённую
попытку; история входного корпуса остаётся источником данных.

`precheck` этого профиля проверяет небольшой отбрасываемый пример на сохранённой
публичной истории. Это smoke корректности, а не прогноз скорости всего корпуса.

Профиль `training.mode = frozen_heads_step_kl` обучает только головы выбора
действий на закреплённом корпусе. Команды `precheck`, `train`, `evaluate`, `resume`
и предстартовый `bench_rl.py comparison` используются так же, как для
полнокорпусного обучения. `learning.jsonl` содержит нулевую точку и диагностику
после каждого принятого шага: предпочтения внутри пар, выбор лучшего действия
среди всех легальных команд и изменения решений на опорах. Группы
`alternative` и `parent_confirmation` разделяют усвоение альтернатив и
подтверждение исходного выбора. `training.json` и progress JSON показывают
последнее принятое состояние этих метрик; это дополнение к боевой оценке.
Предстартовая сводка показывает время подготовки, шага и полного KL с
диагностикой и прогноз для заданных `training.max_accepted_steps` и
`training.max_retries`.

Для CONFIG с `fit_continuation` используйте `precheck`, затем `resume`:
продолжение записывается в отдельный output, сохраняя Adam, RNG, счётчики,
диагностику и использованную часть общего бюджета. `precheck` проверяет исходное
состояние, совместимость сохранённого корпуса и ключ оценки родителя.
Можно задать другой лимит `training_seconds` и больше принятых шагов при
тех же настройках шага, KL и повторных попыток.
`fit-lineage.json` закрепляет происхождение, `reused-parent.json` подтверждает
переиспользование родительского Development. Итоговая оценка выполняется для
последнего принятого кандидата. При несовместимом evaluation cache команда
сообщает ошибку; родитель автоматически не переоценивается.

Для протокола `rl01.action_value` команды `precheck`, `prepare`, `train`,
`evaluate`, `resume` работают исключительно с сохранёнными развилками и исходами.
`collect` не поддерживается. `precheck` требует отдельный OUT и выполняет
отбрасываемый smoke; `train` обучает голову, `evaluate` подтверждает закреплённого
кандидата, `resume` выполняет незавершённые фазы. Бои не запускаются.

`split.json` содержит разбиение миров. `protocol.world_counts` задаёт размеры
training, selection и confirmation и должен покрывать все уникальные миры
корпуса. `seed` задаёт порядок миров и перемешивание при обучении.
`training.json` содержит эпохи, время и лучший selection G.
`selected.json` закрепляет кандидата. `result.json` и `report.md`
показывают confirmation G, CI95 и выполнение критерия перехода к проверке
политики. `confirmation-predictions.pt` содержит logits, вероятности `probs`,
разность `delta_prob = p_alternative - p_parent` и фактический выбор для каждой
развилки. Подготовка и confirmation измеряются отдельно от мягкого лимита
учебного цикла; начатая эпоха завершается вместе с selection.
Состояние сохраняется на границах эпох; после сбоя незавершённая эпоха
повторяется от последней сохранённой границы.

Для `rl01.turn_comparison_fit` используются сохранённые исходные и подтверждающие
доигрывания. `prepare` создаёт `dataset.json`, `train` обучает кандидата,
`resume` продолжает только fit. Development запускается отдельной командой
`evaluate`; Additional отключён. `precheck --out OUT` выполняет отбрасываемый
smoke на реальных историях и опорах, не прогнозирует скорость полного обучения.
Параметры исполнения `--history-microbatch N` и `--[no-]compile` задаются до fit
и сохраняются для resume. `training.positions_per_step` задаёт число исходных
позиций на шаг Adam, `training.max_epochs` задаёт предел числа эпох.
`training.json`, `attempts.jsonl` и `steps.jsonl` показывают принятые шаги,
отклонения, LR, парный loss, полный KL и изменения argmax на опорах.
Бюджет включает подготовку и проверки; после начатой попытки завершаются
контроль KL и сохранение, а превышение записывается в `overshoot_seconds`.
При отсутствии принятых шагов кандидат совпадает с родителем.
При совпадении политики с родителем `evaluate` пропускает оба Development
и записывает `skipped_identical_policy` в `result.json`.
Перед запуском на выбранном хосте измерьте штатный размер шага отдельно от smoke:

```powershell
python tools/bench_rl.py comparison --config CONFIG --out OUT --warmup 1 --steps 1 --trace-steps 1 --history-microbatch 2
```

`benchmark.json` разделяет подготовку, прогрев, шаг, полный KL и рабочие сохранения,
показывает прогноз для принятого объёма и сценария повторных попыток. Все веса
замера отбрасываются; научный каталог не изменяется.

### Измерение CPU/GPU сравнительного цикла

Для `rl01.turn_comparison_collection` доступны `prepare`, `precheck`, `collect`
и `resume`. `prepare` закрепляет список в `positions.json`; `selection.json`
показывает число подходящих и выбранных позиций. `precheck --out OUT` выполняет
короткий отбрасываемый smoke. `--workers W` задаёт число воркеров в smoke и
сборе. `collect` запускает продолжения,
`resume` достраивает незавершённые задачи с теми же seeds. Обучение не запускается.
Прогресс показывает завершённые продолжения и целевой объём. `result.json`
содержит W/D/L вариантов, парные разницы с argmax и разнообразие последовательностей;
истории изменяемого хода находятся в `turn-history.json` каждого продолжения.

Конфиг с `confirmation` проверяет закреплённые sampling-варианты против argmax
на дополнительных парных seeds. Команды те же; `prepare` сохраняет список задач
и трактовку результата в `analysis-plan.json`. `continuations-summary.json`
содержит исходы, seeds, хеши команд и состояния при передаче хода либо завершении
боя; `position-results.json` — парные результаты позиций. `result.json` показывает
среднюю разницу побед, CI95 по мирам и описательные количества положительных,
нулевых и отрицательных эффектов. Обучение по результатам не запускается.

```powershell
python tools/bench_rl.py comparison --run RUN --config CONFIG --out OUT --warmup 1 --steps 3
python tools/bench_rl.py comparison-collection --run RUN --config CONFIG --out OUT --worker-counts 6,8,12,16 --forks 64 --seconds 300
```

Для предстартового замера полного корпуса используйте его CONFIG и
`--warmup 1 --steps 1`. Сводка содержит время полного шага и полного KL,
подготовку, сохранение и прогноз относительно тренировочного бюджета.
Измеряемые обновления отбрасываются. Этот замер выполняется отдельно от `precheck`.

Вход — сохранённые training-истории (или завершённый сбор для
`comparison-collection`); OUT должен быть пустым и вне RUN. Выход —
`summary.json`, телеметрия CPU/NVML и Kineto-трассы (или `recommended-runtime.json`
и прогноз развилок за бюджет для `comparison-collection`).
Для замера сравнительного цикла `--cases N` ограничивает число исходных
матчей, а `--workers W`, `--max-batch N` и `--wait-seconds S` задают runtime
параллельных матчей и inference.
В сводке `steady_resources` показывает загрузку в рабочем окне,
`drain_resources` — при доигрывании хвоста, `resources` — за обе части.
Сравнивайте cont/s с загрузкой именно рабочего окна.
`--warmup-measured-work` выполняет полный проход измеряемых развилок вне замера,
затем повторяет ту же работу с прогретым пулом и inference.
Для сравнения GPU-сервисов задайте `--inference-services N`:
`0` использует поток в основном процессе, `1` и более — отдельные процессы.
`--worker-counts` задаёт общее число CPU-воркеров, а не число на сервис.
Каждый сервис загружает модели; следите за суммарным расходом VRAM.
Флаг `--prefix-states` у `comparison-collection` проверяет восстановление
памяти ветвей без повторных запросов префикса. Сводка показывает число
`prefix_observations_skipped` и `prefix_source_prefills`; полученный runtime
сохраняет выбранный режим. При сборе его можно задать через runtime-файл или
`run_comparison.py collect --prefix-states` (`--no-prefix-states` отключает).

## CPU/GPU-инференс оценок

```powershell
python tools/run_ppo.py run --config CONFIG --evaluation-device cuda
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
В новом смешанном PPO число воркеров по умолчанию определяется доступными CPU
с учётом affinity и квоты Linux. Один логический CPU оставляется координатору,
число воркеров ограничено диапазоном 1–24. Явный `--evaluation-workers`
позволяет выбрать другое число.
Одновременно batch ограничен числом workers. Оценки используют FP32 без AMP/TF32.
На Windows CUDA-командам нужен активированный MSVC environment (`vcvars64.bat`).
После ошибки повторите оценку с тем же runtime: завершённые матчи сохраняются.

## Производительность и GPU

```powershell
python tools/ctl_rl.py performance --run RUN --last 10 --json
python tools/ctl_rl.py overhead --run RUN --out OUT --reference-start START_UPDATE --reference-stop STOP_UPDATE --through-update UPDATE --deployment DEPLOYMENT
```

`performance` читает законченные updates; без `--last` — весь RUN.
Для записи результата перенаправьте вывод `performance --json` в файл.
`overhead` сохраняет разбор затрат
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

Окна GPU Core Load взвешивают выборки по времени. При наличии `phases.jsonl`
учитываются точные границы фаз. Окна прерываются при смене фазы, rollout или
сессии и при пропусках телеметрии.

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
python tools/bench_rl.py replay --run RUN --warmup 1 --repeats 3 --out OUT
python tools/bench_rl.py ppo-update --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --minibatch-tokens 128 --out BASE
python tools/bench_rl.py ppo-update --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --minibatch-tokens 128 --baseline-result BASE/benchmark.json --out OUT
python tools/bench_rl.py selfplay-profile --config CONFIG --checkpoint PARENT --rollout ROLLOUT --episode-limit 150 --out OUT
python tools/bench_rl.py full-cycle --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --out OUT --repeats 3 --device cuda
python tools/bench_rl.py bc --config CONFIG --checkpoint CHECKPOINT --run CORPUS_RUN --scratch-dir .tmp/bc-bench --warmup 3 --steps 5 --trace-steps 3 --verify-parity --parity-steps 3 --out OUT
python tools/bench_rl.py evaluation --run RUN --checkpoint CHECKPOINT --out OUT --cases 2 --workers 4 --warmup 3 --steps 5 --trace-steps 3
python tools/diagnose_cp_replay.py --rollout ROLLOUT --episode EPISODE --trace-from COMMAND --out OUT
```

| Режим | Что измеряет |
|---|---|
| `replay` | Время повторного прогона сохранённого rollout с фиксированными весами и сверкой метрик между повторами |
| `ppo-update` | Одна отбрасываемая эпоха PPO на сохранённом rollout с переиспользуемым эталоном паритета и времени |
| `selfplay-profile` | Смешанный сбор фиксированных боёв и отбрасываемый PPO с двумя частотами полного KL, NVML и три короткие трассы Kineto |
| `full-cycle` | Сбор и отбрасываемый PPO update; нужен PPO checkpoint с Adam |
| `bc` | Прогретые учебные шаги; `--verify-parity` добавляет сверку с baseline |
| `evaluation` | Матчи и inference выбранного checkpoint на сохранённых случаях |
| `diagnose_cp_replay.py` | Повтор решений CP; `--trace-from` требует trace-enabled native library |

`--warmup` исключается из замера, `--steps` ограничивает число учебных шагов,
`--trace-steps` сохраняет короткую Kineto-трассу и сводку ядер. Для replay
`--episode-limit N` выбирает первые N эпизодов, а `--repeats` задаёт число
замеров. Результаты записываются в OUT.

`ppo-update` запускайте до и после правки на той же машине с checkpoint,
содержащим Adam, и rollout, проходящим behavior precheck. Для короткого
эталона используйте небольшой сохранённый rollout или `--episode-limit N`.
Удерживайте одинаковыми входы, настройки батча и seed. Пример выше
воспроизводит размер minibatch короткого аудита на 1 107 решениях.

`benchmark.json` хранит по шагам отпечатки logits, градиентов до clipping
и Adam, численные метрики loss и KL, включая `k3` в режиме epoch-or-spike,
и решения о принятии. При расхождении команда перечисляет поля и возвращает
код 1. `--check-rollback` добавляет принудительное отклонение первого шага
и сверку восстановленного состояния. Включайте флаг в обеих ветвях при
проверке изменений, которые затрагивают rollback.

Для A/B резидентного PPO replay используйте `--no-resident-replay-indices`
на исходном замере, затем тот же CONFIG и rollout с `--baseline-result`.

Время берётся из отдельных прогретых повторов полного update. Прогоны с
новой компиляцией исключаются из медианы. Старый эталон годится для
первичного отбора. Если разница сравнима с разбросом, переснимите обе ветви
в одной сессии. Для решений о throughput полного цикла используйте штатный
объём работы и профиль затронутой фазы.

PPO на CUDA с BF16 по умолчанию переиспользует spatial CUDA Graph для
no-grad блока 32 × 256 в режиме `default`. Для замера обычного compiled
forward в этом профиле задайте `performance.ppo_spatial_cuda_graph=false`
в копии CONFIG.

## Профилирование collection и подбор runtime

```powershell
python tools/profile_rl_runtime.py cpu --rollout ROLLOUT --episodes 8 --repeat 3 --python-profile --out OUT
python tools/profile_rl_runtime.py collection --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --episodes 32 --envs 32 --workers 8 --reply-size 2 --repeat 3 --out OUT
python tools/tune_rl_collection.py --scheduler streaming --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --episodes 80 --budget-seconds 600 --out OUT
```

Вход — CONFIG, CHECKPOINT и сохранённые случаи ROLLOUT. Выход подбора —
`recommended-runtime.json`; передайте его runner через `--runtime-config`.
В профилировании смешанного сбора с ненулевой долей архива используются два
архивных идентификатора с весами CHECKPOINT, чтобы измерять смешанные волны.
`--no-record-hidden` измеряет сбор без выгрузки скрытых состояний, как в
обучении after-target-BC PPO. Трасса по умолчанию включает их для диагностики.
При автоматическом подборе для self-play `M` — целая часть доступного
CPU-бюджета с учётом affinity и квоты, минимум один. Тюнер измеряет `M`
и `2M` на одинаковых случаях и подтверждает лучшую конфигурацию каждого
числа воркеров повтором. Он выбирает `2M` при росте медианы решений learner
в секунду минимум на 10%. Для ручного подбора используйте `--candidates`.
Автоматическому сравнению нужно не менее `2M` случаев.
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
`--min-vram-gib` задаёт минимальную VRAM по данным оффера.
Для поиска по нескольким моделям GPU повторите `--gpu-name` для каждой модели,
например `--gpu-name "RTX 4090" --gpu-name "RTX 5090"`. Указанные модели
заменяют модель профиля. Остальные фильтры действуют для каждой модели.

### Подготовка, tuning и запуск

Нужны чистое рабочее дерево и HEAD в origin. CARRIER — сохранённый rollout
с открытыми случаями для tuning after-target-BC PPO.

```powershell
git push origin HEAD
python tools/remote_rl.py package --config CONFIG --out DEPLOYMENT --instance INSTANCE_JSON --extra-input CARRIER
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode tune
python tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode run --evaluation-device cuda
python tools/ctl_rl.py progress --deployment DEPLOYMENT --watch --interval 900
```

`package` собирает входы и описание запуска в DEPLOYMENT. `launch` готовит
окружение и запускает процесс; при поддержке протокола `tune` подбирает runtime.
Результаты хранятся на сервере до `pull`.

Для отдельного профилирования смешанного PPO используйте `launch --mode profile`
с тем же `--extra-input CARRIER` на этапе `package`. После `pull --keep-instance`
в `DEPLOYMENT/profiles/` доступны результаты и исходные сжатые Kineto-трассы.
Профилирование не создаёт состояние обучения и не заменяет tuning для `run`.

Для CPv2 используйте отдельный пилотный конфиг и носитель тренировочных случаев.
После `package` с `--extra-input CARRIER` сначала выполните и выгрузите `tune`,
затем запустите три rollout и выгрузите их через `pull --keep-instance`:

```powershell
python tools/run_ppo.py prepare-carrier --config configs/rl-ppo-c2pa-h60-cpv2-pilot.json --out CARRIER.pt
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode run --diagnostic-cpv2
python tools/run_ppo.py pilot-report --config configs/rl-ppo-c2pa-h60-cpv2-pilot.json --out PILOT_RUN
```

Пилот ведёт сбор и PPO на тренировочных мирах и отключает Development и Additional.
Для C2PA-pre-GELU используйте `configs/rl-ppo-c2pa-pre-gelu-h60-cpv2-pilot.json`.
`pilot-report` проверяет attention на сохранённых позициях R3 по чекпойнту после
PPO R3 и возвращает число принятых шагов и хеши входов в `R3_post`.
В отчётах исторического C2PA метрики attention находятся в `R3_pre`, там же
доступна предложенная поправка LR и эпох для отдельного повторного пилота.

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
Повтор `launch` активного job присоединяется к нему. Для переноса незавершённого запуска на другой хост сохраняйте RUN/execution.json,
checkpoints и входные данные; checkout должен соответствовать записанному commit.

Для трёхстороннего PPO `launch --mode resume --continue-after-dual-target`
восстанавливает сохранённую границу победной остановки, если первый Additional
дал 188–191 победу по более слабой из двух серий против ботов и H2H неотрицателен.
После переноса на другой инстанс используйте тот же commit и сохранённые нативные
библиотеки. Отбор новой модели использует вехи после исходной границы.

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
python tools/ctl_rl.py restore-evaluations --run RUN --match RELATIVE_MATCH_DIR --out OUT
```

`archive-evaluations` проверяет архив и заменяет `evaluation/` на
`evaluation.storage/`. `restore-evaluations` возвращает распакованные файлы для
просмотра или resume. Формат — PPMd, словарь 256 МиБ, order 32, solid 4 ГиБ.
С `--match` команда выгружает один каталог боя вместе с `replay.json` в `OUT`.
Укажите путь относительно `evaluation/`. Полный архив остаётся на месте.
Для paused запуска архивирование выполняется отдельной командой после остановки.

`package` и `launch` упаковывают большие входные наборы JSON. На обеих машинах
нужен `7zz`, `7z` или `7za`. В новых deployment после завершения прогона сервер
выполняет аудит и собирает `evaluation.storage/`. `pull` получает готовый архив,
чекпойнты, отчёты и телеметрию через проверяемое зеркалирование `remote-python-ssh`.
При сбое передачи повторите `pull`, чтобы забрать тот же пакет. Подтверждение
доставки и аудит находятся в DEPLOYMENT. Для отдельного повторного аудита
используйте `ctl_rl.py audit`.

### Жизненный цикл deployment

`status` показывает состояние удалённого процесса. `pull` ждёт завершения job,
скачивает результаты в RUN и сохраняет аудит. Для paused, прерванных и старых job
аудит выполняется локально после выгрузки. По умолчанию инстанс сохраняется.
`--delete-instance` удаляет его после успешной выгрузки и проверки.
