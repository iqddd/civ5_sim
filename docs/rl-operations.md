# Работа с RL-запусками

Команды выполняются из корня репозитория в окружении опыта. `RUN`, `CONFIG`, `CHECKPOINT`, `ROLLOUT` и
`OUT` ниже — подставляемые пути. Условия обучения, parent и критерии выбора
модели ищите в [журнале](experiment-ledger.md), устройство паспортов — в
[архитектуре](rl-architecture.md#паспорта-исходники-и-resume).

Текущие BC/PPO runners исполняют фазы своих протоколов.
Для генерации корпуса служит отдельная фаза без обучения:

```powershell
python tools/run_cp_imitation.py prepare-corpus --config configs/rl-v1.json --out CORPUS_OUT --game GAME_DIRECTORY --workers 4
```

Она использует `corpus.parent_worlds` и параметры генерации из конфигурации,
не запускает BC/PPO и не создаёт модель. Остальные фазы BC используют уже
подготовленный corpus manifest.

## Анализ сохранённого запуска

Для повторного просмотра открытой серии на тех же сохранённых картах,
расстановках и seeds с новым checkpoint:

```powershell
python tools/ctl_rl.py play-cases --manifest SHOWCASE_MANIFEST --checkpoint CHECKPOINT --out NEW_MATCH_DIRECTORY
python tools/view_combat.py NEW_MATCH_DIRECTORY --port 8767
```

`play-cases` принимает manifest серии `civ5rl.showcase` только с открытыми
случаями и новым пустым OUT. Использует прежние правила CP и seeds, argmax,
CUDA deterministic FP32, 8 workers/max batch 32/wait 0,001 с; проверяет каждый
сохранённый replay и неизменность checkpoint. Это просмотр, не Development
или Additional. Файлы исходной серии не изменяются.

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

### Замена команды в policy–CP реплее

```powershell
python tools/ctl_rl.py branch-replay --source MATCH_DIRECTORY --command 30 --action move --unit 8 --target 4 3 --check-only
python tools/ctl_rl.py branch-replay --source MATCH_DIRECTORY --command 30 --action move --unit 8 --target 4 3 --checkpoint CHECKPOINT --out NEW_MATCH_DIRECTORY
```

Номер команды — с единицы, координаты — с нуля. Для `skip`/`fortify` цель не нужна.
`--continue-to-round 90` вместо замены продолжает полный сохранённый префикс
до увеличенного лимита; `--command` не нужен. Model/CP reset получают исходный
горизонт, поэтому у C2P после60 раундов elapsed=1 и remaining=0 (clamp),
а реальные state.turn,таймер контактов и хронология продолжаются.
`--forbid-move-to X Y` вместо замены маскирует цели move/swap на заданную
клетку для всех юнитов модели с `--command` до конца текущего хода. На
следующих ходах ограничение снимается; промежуточные клетки пути не запрещены.
Forward получает исходный список легальных действий, маска применяется к logits
при выборе argmax. Ограничение и номер хода записываются в experiment реплея.
Вместо `--action/--unit/--target` можно указать `--actions FILE.json`: непустой
JSON-массив команд `{ "kind": "move", "unit": 8, "target": 39 }`, где target —
линейный индекс клетки `y*width+x`. Вся последовательность исполняется подряд
начиная с `--command`, затем управление возвращается модели. Каждая команда
проверяется до загрузки модели; переход на сторону CP внутри последовательности запрещён.
`--check-only` проверяет точный префикс и законность, не загружает модель и не
пишет результаты. Исполнение требует новый OUT, `case.json` рядом с исходным
реплеем и checkpoint той же политики. По умолчанию CUDA FP32 argmax без AMP/TF32;
`--device cpu` задаётся явно. Сохраняются seeds и режим сброса хода источника.
Память модели восстанавливается по исполненным командам, включая прежние ручные
замены; CP исполняет и сверяет свои решения префикса, сохраняя координатор.
Несовпадение результатов/hash префикса или незаконная замена блокируют ветку.
Выход содержит replay/result/final, case, происхождение и проверку реплея.
Модуль относится к reporting и не меняет training execution identity. Команда
не открывает браузер; новая ветка в каталоге серии появляется после обновления панели.

## Прогресс процесса, EMA и ETA

```powershell
python tools/ctl_rl.py progress --run RUN
python tools/ctl_rl.py progress --run RUN --watch --interval 900
python tools/ctl_rl.py progress --deployment DEPLOYMENT --watch --json
python tools/ctl_rl.py progress --run RUN --alpha 0.1 --exclude-compilation
```

У `tools/run_ppo.py` и `tools/run_cp_imitation.py` для исполняемых фаз
обучения/оценки автоматически работает `ProcessProgress`: каждые 30 секунд
и при выходе пишет атомарный `RUN/derived/progress/latest.json`. Это отдельный
наблюдатель процесса, включая запуск через remote worker. Он читает факты из
ledger/state/телеметрии; не изменяет model/Adam/RNG, config, checkpoints или
научный status. Ошибка наблюдателя выводится в stderr и не останавливает обучение.
`session.json` хранит границу текущего процесса; оба файла — заменяемые производные
данные, не входы resume. При аварийном завершении без cleanup heartbeat стареет;
свежее чтение артефактов само по себе не доказывает, что процесс жив.

Без `--watch` команда печатает один срез; с ним повторяет запрос через 900 секунд
по умолчанию. `Ctrl+C` останавливает только просмотр. SSH-ошибка при просмотре
не останавливает удалённый процесс; следующий запрос повторяется через interval.
`--deployment` рассчитывает небольшой срез на удалённом хосте через SSH,
не скачивая checkpoints/ledger. Там нужна версия кода с этим слоем. Команда
`remote_rl.py status` также показывает последний записанный `progress`, который
может отставать на интервал записи. Постоянной фоновой синхронизации файлов нет.
После удаления инстанса используйте `--run` на выгруженных артефактах.
Внутренняя запись раз в 30 секунд не меняет политику ожидания агента: после
подтверждённых 2–3 шагов проверять стабильное длительное обучение раз в 15 минут.

EMA (`alpha=0.1`) обновляется один раз на единицу измерения, не на опрос:
PPO — завершённый rollout, решения / приращение active time сбора и обновления;
BC — записанный завершённый шаг, labels / приращение active time. При лимите BC
`max_adam_steps` показываются steps/s и остаток шагов. В старом BC без cumulative
active time доступна только скорость по таймерам шагов; её ограничение указано,
а попадание продолжающегося обучения в active budget остаётся неизвестным.
Новые target-scale BC runners записывают cumulative active time как низкоуровневый
факт. Данные текущего PPO rollout до commit показываются отдельно как collected;
они не вычитаются из остатка завершённой работы.

На каждом старте/resume rate и калибровка длительности оценок начинаются заново,
чтобы не смешивать разные машины и компиляционные запуски. До первого измерения
скорость/ETA неизвестны; при чтении исторического запуска без session смешение
истории явно отмечается. Уже частично исполненный до resume PPO rollout не
используется как sample нового процесса: его PPO-only завершение не характеризует
полный цикл сбора и обучения. `--exclude-compilation` вычитает только записанные Dynamo
callbacks из интервалов samples. Это сценарий без этих callbacks, а не прогноз
исчезновения всех накладных расходов или будущих падений.

Work ETA, обязательные оставшиеся фазы и условные оценки показаны отдельно.
Активный бюджет учитывает только те фазы, которые его расходуют; Development
и Additional останавливают active clock, поэтому не входят в его `fits`.
Неизвестная обязательная фаза делает total ETA неизвестным. Незавершённая работа
считается целиком; прогноз не вычитает уже потраченную часть текущего rollout/оценки.
У paused/error среза ETA означает продолжение после resume; календарное окончание
не обещается. После early stop будущие недостигнутые milestones не прогнозируются.

Специализация after-target-BC PPO считает Development по milestones и последней
изменённой модели; Additional-256 двух моделей остаётся условным до выбора/gate.
Его длительность масштабируется из Development по фактическому числу матчей.
Для оценок берётся медиана последних трёх измерений текущего процесса длительностью
более 10 секунд, чтобы не использовать короткие cache hits. Target-scale BC знает
evaluate_epochs; C2 BC — четыре reference-модели, milestone/latest candidates и
offline metrics после эпох. У других протоколов непредставленные фазы остаются
неизвестными, а не получают выдуманный нулевой ETA. Короткий report/post-processing
и неотдельно измеренные diagnostics могут давать дополнительную погрешность.

## Пауза и закрытие

```powershell
python tools/ctl_rl.py request-pause --run RUN --after-update 12
python tools/ctl_rl.py request-pause --run RUN --after-epoch 10
python tools/ctl_rl.py wait-and-finalize --run RUN --out OUT --timeout 600
python tools/ctl_rl.py close --run RUN --reason "Research stopped at saved boundary"
python tools/ctl_rl.py extend-budget --run RUN --additional-seconds 3600 --reason "User requested one extra active hour"
python tools/remote_rl.py extend-budget --deployment DEPLOYMENT --additional-seconds 3600 --reason "User requested one extra active hour" --request-id extra-hour-1
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

`extend-budget` поддерживает работающий либо возобновляемый after-target-BC PPO
и target-scale BC, включая more-data/C2. Используется отдельный `budget.lock`,
поэтому writer lock обучения не мешает запросу. Согласованные секунды добавляются
в `active-budget.json` с `run_id`, исходным лимитом и историей запросов; config,
checkpoints, model/Adam/RNG и потраченное время не переписываются. Лимит решений
или эпох не меняется. Завершённое обучение, закрытый запуск и уже установленный
stop_reason продлением не возобновляются; другим runners команда явно отказывает.

`--request-id` позволяет безопасно повторить запрос после обрыва SSH: тот же ID,
число секунд и причина не создают второй прибавки; изменение содержания под тем
же ID отклоняется. Удалённая команда печатает ID перед SSH, в том числе когда он
сгенерирован автоматически. Она меняет файл на удалённом хосте без pull/redeploy.

`requested_revision` означает запись запроса, а `applied`/`applied_revision` —
подтверждение runner в `budget-applied.json`. Runner читает изменения не чаще
раза в секунду на безопасных проверках и применяет новый deadline внутри PPO
update/collection или BC эпохи; GPU kernel и компиляцию не прерывает. Во время
оценки active clock стоит, подтверждение может ждать следующей проверки обучения.
Запрос на самом завершении может остаться pending: запись запроса сама по себе
не гарантирует продление уже завершившегося процесса. Status/progress показывают
обе ревизии и эффективный лимит. Старому процессу без этих hooks нужна штатная
пауза и новая версия runner; смена кода оформляется через `source-revisions/`.

## CPU/GPU-инференс оценок

```powershell
python tools/run_ppo.py run --config CONFIG --evaluation-device auto --evaluation-workers 8 --evaluation-max-batch 32 --evaluation-wait-seconds 0.001
python tools/run_cp_imitation.py evaluate --config CONFIG --evaluation-device cuda
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode resume --evaluation-device cpu
python tools/bench_rl.py evaluation --run RUN --checkpoint CHECKPOINT --out OUT --cases 2 --workers 4 --warmup 3 --steps 5 --trace-steps 3
```

Общий runtime подключён к Development/Additional target-scale BC (включая
more-data/C2), target-scale PPO и after-target-BC PPO. Остальные исторические
runners сохраняют прежний путь и отказывают явно переданным runtime-флагам.
У нового запуска по умолчанию `auto`: CUDA при наличии, иначе CPU. `cuda` без
CUDA выдаёт ошибку. CP–CP mirror исполняется на CPU. Первый выбор сохраняется
в `evaluation-runtime.json`; новый run record отмечает default `auto`. У
исторического run record без этого признака выбирается CPU, даже если evaluations
ещё находятся в архиве. Явные CLI-флаги имеют приоритет над сохранённым выбором.
Параметры не добавляются в научный config. По умолчанию workers берётся из него,
max batch = 32, ожидание объединения готовых запросов = 1 мс. Remote launch
передаёт те же флаги worker и сохраняет их в deployment.

На GPU одна оценочная модель обслуживает запросы CPU-воркеров в основном
процессе. Воркеры исполняют обычные симулятор/CP и пишут стандартные replay,
final/result/case/metrics; CUDA-моделей в каждом воркере нет. Энкодер, RNG и GRU
разделены по боям; forced actions обновляют память, reset обнуляет её. Оценочная
модель отдельна от обучаемой; используется общий аллокатор основного процесса.
Режим — deterministic FP32, без AMP/TF32 обучения; настройки вызывающего кода
восстанавливаются после оценки. Побитовое совпадение CPU и GPU не обещается.

Backend, численный режим, GPU, параметры пакетирования и hash inference service
входят в ключ evaluation cache и metadata результата. CPU и GPU не разделяют
результаты по одному только policy hash; явная смена режима создаёт отдельный
каталог результатов. Cross-run reuse проверяет runtime, CPU archive baseline
не используется как GPU baseline. При ошибке нет молчаливого CPU fallback:
завершённые матчи остаются, повтор тем же runtime пропускает их по metrics.
EMA длительности оценок не смешивает разные runtime identities.

`bench_rl.py evaluation` выполняет ограниченное число сохранённых validation
случаев на CPU и GPU, проверяет созданные replay и неизменность исходных
checkpoints/ledger. Отдельно измеряет несколько прогретых inference batches,
сверяет CPU/GPU logits/hidden на одинаковой публичной истории и при запросе
сохраняет короткую Kineto-трассу. Для full-match wall учитываются также startup,
CP, IPC и запись файлов; whole-pass NVML не выдаётся за чистый GPU-интервал.
Короткий benchmark не доказывает пропускную способность полного Development-512.

## Производительность и GPU

```powershell
python tools/ctl_rl.py performance --run RUN --last 10 --json
python tools/ctl_rl.py overhead --run RUN --out RUN/derived/overhead --reference-start 49 --reference-stop 59 --through-update 76 --deployment DEPLOYMENT
```

Без `--last` анализируется весь запуск. Учитываются только законченные строки
`updates.jsonl`; текущий неполный rollout не включается. Reporter работает без
Torch/GPU и показывает timers, coverage, фазовые NVML means, непрерывные
60-секундные окна и наблюдённые GPU-фазы от десяти минут. Запись файлов
по запросу требует `--output`; обычный вызов читает сохранённую телеметрию.

| Показатель | Смысл |
|---|---|
| `collection` | Wall time сбора, записанный collector |
| `ppo_recorded` | В текущем PPO `ppo_seconds` включает создание backend и precheck |
| `ppo_operation` | Внешний интервал вызова `update()` в runner, включая backend и precheck |
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

`overhead` разделяет active time на collection, Dynamo callbacks внутри PPO,
PPO без этих callbacks и явно неатрибутированный остаток. KL/backward/cache —
вложенные таймеры, складывать их с этим балансом нельзя. Reference range
задаётся явно; масштабирование по решениям — описательное сравнение, не
доказательство оптимальной скорости. `--deployment` можно повторять для
истории процессов: паузы между ними не приравниваются к расходу GPU или оплате.
Phase telemetry может включать неудачные попытки того же rollout, поэтому её
итоги не заменяют таймеры committed updates. Текущий runner передаёт monitor
в PPO: preparation/precheck отмечаются отдельно; старые данные от этого
автоматически не получают более точную разбивку.

`integrate_gpu_interval` принимает секунды от `ResourceMonitor.start()`,
не UTC/абсолютный `perf_counter`. Coverage знаменует весь запрошенный интервал,
включая отфильтрованные фазы. У after-target-BC report также формируются
`performance.json/.md` и `performance-last10.json/.md`.

При накопительном CUDA OOM сначала различают рост живых тензоров, резерв
аллокатора и удержание CUDA Graphs: сопоставляют driver-free, allocated,
reserved/inactive split на одинаковых границах updates. Один большой reserved
не доказывает утечку, а устойчивый allocated не исключает расход графов.
`empty_cache()` не освобождает живые приватные пулы графов; отключение проверки
резерва не устраняет источник расхода памяти.

Если подтверждённого исправления нет, согласованные плановые перезапуски
выполняют через `request-pause --after-update`, дожидаются `paused`, затем
`pull --keep-instance` и `launch --mode resume` на том же deployment. Обычная
пауза не добавляет final evaluation. Сохраняются model/Adam/RNG, счётчики и
потраченный active budget; новый процесс заново прогревает и компилирует графы.
Интервал выбирают с запасом по наблюдённому тренду VRAM, не дожидаясь OOM.
Не сбрасывать внутренние CUDA Graphs/компилятор в работающем шаге.

After-target-BC PPO автоматически пишет `telemetry/cuda-memory.jsonl` перед
collection, перед оптимизацией, после существующего `gc.collect()` и после
оценки. Это чтение host-счётчиков аллокатора, включая private pools по pool ID,
retry/OOM/device allocation counters, формы и pool IDs GRU-кэша. Счётчик записей
Inductor и его pool ID доступны как необязательные version-specific metadata;
это все графы устройства в текущем потоке, не только один spatial policy.
Счётчики накоплены за процесс, не сбрасываются на каждом update. NVML-данные
берутся из последней фоновой выборки с её собственным временем, не синхронно
с границей. Не складывать allocated/reserved/private pools как независимые
объёмы. Неподдерживаемые поля остаются null/status unavailable.

Нормальные записи не запускают CUDA kernels, `synchronize()`, `.item()`,
`empty_cache()` или memory history; стоимость чтения и записи мала, но не нулевая.
`sampling_seconds` включает сбор данных, а не последующую сериализацию/запись.
При CUDA OOM отдельно сохраняется сжатый layout аллокатора
`telemetry/cuda-oom-SESSION-TIME.json.gz`, без предварительного включения истории
каждой аллокации и её стеков. Эта более дорогая операция выполняется только
после сбоя; ошибка диагностики не заменяет исходное исключение.

## Replay, полный цикл и диагностика

При сравнении новых attention/backward ядер отдельно измеряют оператор и полный
PPO update. Для сравнения updates прогревают весь путь (не только backward):
precheck, KL replay, Adam и допустимые CUDA Graphs. Между вариантами восстанавливают
одни и те же model/Adam/RNG и behavior labels; измерения чередуют. Одиночный update
с впервые исполняемыми ветвями не является оценкой устойчивого ускорения.
Отбрасываемые диагностические фрагменты из публичных наблюдений проверяют путь
вычисления, но не дают прогноз collection throughput или силу policy.
Принятие нового численного режима следует правилам
[корректности и replay](rl-architecture.md#численная-корректность-и-воспроизводимость-оптимизаций):
градиенты сравнивают с допусками, behavior replay gate сохраняют. Закреплённый
CUDA/FP32 runtime оценок продолжения не заимствует precision или TF32 из обучения.

```powershell
python tools/bench_rl.py replay --run RUN --compare-git-rev HEAD --out OUT
python tools/bench_rl.py replay --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --device cpu --episode-limit 1 --warmup 1 --repeats 1 --out OUT
python tools/bench_rl.py full-cycle --config CONFIG --checkpoint CHECKPOINT --rollout ROLLOUT --out OUT --repeats 3 --device cuda
python tools/bench_rl.py bc --config CONFIG --checkpoint CHECKPOINT --run CORPUS_RUN --scratch-dir .tmp/bc-bench --warmup 3 --steps 5 --trace-steps 3 --verify-parity --parity-steps 3 --out OUT
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
`bc` прогревает компиляцию вне замеряемого окна, затем измеряет итеративное BC-обучение по NVML (`60s` и интеграл интервала), при `--trace-steps` сохраняет Chrome-трассу Kineto CPU+CUDA и `*-trace-summary.json`, а при `--verify-parity` сверяет CE, норму градиента и обновлённые веса с синхронным baseline-циклом, не модифицируя каталог научного прогона. На Windows при включённом `torch.compile` команду запускают с активированным MSVC (`vcvars64.bat`, чтобы `cl.exe` был в `PATH`). Для предстартовой проверки достаточно короткого прогона (`--warmup 3 --steps 5 --trace-steps 3`): помимо NVML Core Load проверяют `% TDP` и `top_kernels` в сводке трейса (должны доминировать GEMM и Triton-ядра, а не детерминированный `indexing_backward` или `DeviceRadixSort`), после чего сверяют `steps_per_epoch × step_seconds × epochs` с лимитом `training_seconds`. Для запусков PPO (`rl01.ppo_after_target_bc`) предстартовая сверка бюджета (`compute_time_budget_check` в `protocols/ppo_after_target_bc.py`, сохраняемая в `run.json` при `initialize`) дополнительно учитывает превышение номинального числа шагов из-за неполных последних minibatch в каждой эпохе (`+epochs` шагов на каждый rollout при упаковке `pack_minibatches`) и доигрывания начатых эпизодов на границе бюджета, а также число полных replay-проверок KL. Для `whole_rollout_after_epoch_or_minibatch_spike` измеренный прогноз задаётся `expected_kl_checks_per_rollout`; без него ожидаемое попадание в бюджет остаётся неизвестным. Отдельно показывается пессимистичный сценарий: precheck, проверка после каждого шага при защёлкнутой жёлтой зоне и возможная проверка отката. Старый резерв «одна проверка на всплеск» не является верхней границей. Превышение выводится предупреждением с числами, а не автоматическим запретом из-за одного худшего сценария.
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

Выберите научный CONFIG и отдельный пустой DEPLOYMENT. Условия, parent,
milestones и результаты конкретных опытов находятся в [ledger](experiment-ledger.md)
и конфигурациях; эта инструкция задаёт только порядок исполнения.

### Аренда и CPU-допуск

Аренда требует согласованного профиля и фильтров. Используйте штатный
`vastai-instance-manager`; проектный wrapper добавляет CPU-допуск:

```powershell
python tools/remote_rl.py rent --profile 5090 --parallel 2 --cpu-score-min 1250
```

Остальные согласованные фильтры передавайте именованными флагами менеджера.
Из строки `SUCCESS` сохраните отдельный JSON подключения с `instance_id`,
`host` и `port`; этот файл передаётся в `package --instance`.
По умолчанию проекта готовятся два параллельных кандидата; сохраняется один
успешный, остальные удаляются. Для CPU-зависимых фаз, включая PPO collection,
требуется `cpu_cores_effective * (100 * SingleThread / 2067) >= 1250`.
Используется доступная логическая квота конкретного offer, включая дробную,
а не полное число потоков модели CPU. Для одного BC этот фильтр не нужен.

Единственный машинный справочник —
[configs/vast-cpu-ratings.json](../configs/vast-cpu-ratings.json): модель,
Single Thread, источник. Неизвестный CPU останавливает отбор: найдите точную
модель в PassMark Single Thread и добавьте запись перед повтором аренды.
`single_thread: null` означает выполненный поиск без подтверждённого результата;
такая модель не допускается. Не подменяйте Emb обычной моделью или Single Thread
значением Multi Thread. CPU-балл — эвристика отбора; реальная quota/cpuset и
скорость фиксируются на арендованной машине при tuning.

### Подготовка, tuning и запуск

На Windows команды проекта выполняйте через `.venv\Scripts\python.exe`,
менеджер Vast — через системный Python с keyring. `package` требует чистого
рабочего дерева и HEAD, отправленного в origin. Parent checkpoint, manifest,
конечные state/run и только открытые карты упаковываются автоматически.
Для after-target-BC PPO нужен CARRIER — сохранённый rollout с открытыми
расстановками для tuning; его веса и labels не становятся новым training rollout.
Collection benchmark применяет режим физики текущего CONFIG к копии сценариев.

```powershell
git push origin HEAD
.venv\Scripts\python.exe tools/remote_rl.py package --config CONFIG --out DEPLOYMENT --instance .tmp/instance.json --extra-input CARRIER
.venv\Scripts\python.exe tools/remote_rl.py launch --deployment DEPLOYMENT --mode tune
.venv\Scripts\python.exe tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
.venv\Scripts\python.exe tools/remote_rl.py launch --deployment DEPLOYMENT --mode run --evaluation-device cuda --evaluation-workers 8 --evaluation-max-batch 32 --evaluation-wait-seconds 0.001
.venv\Scripts\python.exe tools/ctl_rl.py progress --deployment DEPLOYMENT --watch --interval 900
```

Runtime оценок должен соответствовать CONFIG. Закреплённые значения нельзя
изменить при resume; отсутствие CLI-флага не отменяет закрепление в профиле.
Для другого разрешённого runtime передавайте соответствующие значения.
После запуска короткими проверками подтвердите 2–3 принятых шага Adam,
затем используйте прерываемые ожидания по15минут. Сообщение пользователя
прерывает ожидание, но не процесс обучения.

### Восстановление и завершение

При потере SSH повтор `launch` активного job выполняет attach, не новый запуск.
После завершения или ошибки сначала выгрузите актуальное состояние, затем resume:

```powershell
.venv\Scripts\python.exe tools/remote_rl.py status --deployment DEPLOYMENT
.venv\Scripts\python.exe tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
.venv\Scripts\python.exe tools/remote_rl.py launch --deployment DEPLOYMENT --mode resume
```

Resume сохраняет model/Adam/RNG, потраченный бюджет и счётчики. Новый опыт от
родителя имеет отдельный профиль, output и RNG; не используйте для него resume
исторического каталога. Архивированные локальные оценки перед resume восстановите
командой `restore-evaluations`.

После финальных Development/Additional выполните `pull --keep-instance`.
Проверьте delivery receipt и обязательные секции научного аудита; затем удалите
только привязанный ID:

```powershell
python C:\Users\Roman\.codex\skills\vastai-instance-manager\scripts\vastai_instance.py delete INSTANCE_ID
python C:\Users\Roman\.codex\skills\vastai-instance-manager\scripts\vastai_instance.py inspect INSTANCE_ID
```

`pull --delete-instance` совмещает выгрузку, аудит и удаление, если среда Python
имеет credentials/keyring менеджера. Ошибка доставки или аудита сохраняет инстанс.
Отчёт и результаты остаются в artifacts, краткая запись со ссылками — в ledger.

| Kind | run | tune | resume |
|---|---|---|---|
| `rl01.ppo_after_target_bc` | Да | Да | Да |
| `rl01.ppo_target_scale` | Да | Нет | Да |
| `rl01.ppo_scale_up` | Да | Нет | Нет |
| `rl01.cp_imitation_target_more_data` | Да | Нет | Да |
| `rl01.cp_imitation_target_c2` | Да | Нет | Да |
| `rl01.cp_imitation_target_scale` | Да | Нет | Нет |

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
   SHA-256-зеркалирования `prepare.sync`. Большие JSON-группы передаются пакетами в
   `<remote_root>/transport/inputs/` и проверенно распаковываются worker перед запуском.
   Они не пересекаются с `config["output"]` и не пересылаются обратно при `post-sync`.
3. **Основной вывод опыта**: новые deployments (`archive_transport: 1`) передают
   входное состояние `config["output"]` (для `tune` — `DEPLOYMENT/tuning`) через отдельный
   пакет `DEPLOYMENT/transfers/job-N/upload` ↔ `<remote_root>/transport/job-N/upload`.
   Worker распаковывает его в настоящий каталог опыта до запуска runner.
   `pull` после завершения job отдельно упаковывает и выгружает актуальный научный
   каталог, проверяет и восстанавливает его в `artifacts/rl/<run>`, затем запускает аудит.
   Старые замороженные deployments сохраняют записанное прямое зеркалирование.
4. **Временный кэш (`CIV5RL_SCRATCH_DIR`)**: подготовленные `.npz` файлы BC (`prepared-inputs/`)
   размещаются в `<remote_root>/scratch` вне синхронизируемых каталогов — они сохраняются
   между `resume` на том же хосте и никогда не выгружаются при `post-sync`.

### Передача больших наборов JSON

Завершённые оценки можно хранить на диске в PPMd без распакованного дерева:

```powershell
python tools/ctl_rl.py archive-evaluations --run RUN
python tools/ctl_rl.py restore-evaluations --run RUN
```

Первая команда под writer lock создаёт `RUN/evaluation.storage/json.7z` и manifest
с SHA-256 всех файлов, проверяет полное извлечение и только затем удаляет
`evaluation/`. Параметры: PPMd, 256 МиБ, order 32, solid 4 ГиБ.
Аудит временно извлекает архив и убирает неизменённую копию после проверки.
Для просмотра, сравнения каталогов и возобновления сначала выполните
`restore-evaluations`, после работы — `archive-evaluations`. Восстановление
сохраняет архив; повторное архивирование обновляет его проверенной копией.
Финальный `pull` после успешного аудита запуска со статусом `complete`
автоматически архивирует локальные оценки. Активные и paused запуски
автоматически не архивируются. Исторические научные байты не переписываются.


Перед передачей огромных наборов JSON в любом направлении упаковывать их в **7z,
PPMd, словарь 256 МиБ, размер слова (order) 32, solid-блок 4 ГиБ**:
`7z a -t7z -m0=PPMd:mem=256m:o=32 -ms=4g <archive.7z> <relative-paths>`.
Размер блока здесь относится к solid-сжатию, а не к разбиению архива на тома.
Сохранять исходные относительные пути и байты файлов; после передачи проверить архив,
распаковать и сверить SHA-256 исходных файлов, затем выполнить обычный научный аудит.
Не менять исторические результаты или паспорта ради упаковки.

`civ5rl/deploy/archive.py` автоматически архивирует `.json` и `.jsonl`, когда в
передаваемом дереве **не менее 1000 таких файлов или не менее 64 МиБ их содержимого**.
Остальные файлы, включая checkpoint и уже сжатые `.json.gz`, передаются без повторного
сжатия. На обеих машинах требуется `7zz`, `7z` или `7za`; отсутствие обнаруживается
до начала обучения. Эти настройки относятся к транспорту, не к научному конфигу.

Команды `package`, `launch` и `pull` остаются прежними. Навык проверяет доставку
входного пакета; обратный `post-sync` этого пакета завершает жизненный цикл job,
но сам по себе не подтверждает выгрузку результатов. После него `pull` под writer
lock упаковывает остановленный научный каталог по SSH и скачивает пакет одной
SCP-сессией. Этот этап работает и после аварийного завершения runner; упаковка
не зависит от `finally` тренировочного процесса. Незавершённое применение входного
пакета блокирует восстановление результатов, чтобы не заменить локальную копию
устаревшими либо пустыми удалёнными данными.

Транспорт сверяет SHA-256 полученного manifest с удалённым, SHA-256 архива,
список и пути файлов, затем SHA-256 каждого распакованного файла во временном каталоге.
Только после этого обновляется точное дерево назначения и проверяется итоговая копия.
Token операции и направление защищают от применения старого/входного пакета вместо
результатов; ошибки блокируют отметку `pulled`, resume и удаление инстанса.
Повтор `pull` после сбоя повторяет незавершённую передачу. Подтверждение и manifest
сохраняются в deployment; научные исходники и паспорта не меняются.
Архивы живут вне научного каталога, исходники сохраняются до проверки копии.

### Жизненный цикл deployment

`INSTANCE_JSON` задаёт `instance_id`/`id`, `host`/`SSH_host`, `port`/`SSH_port`; credentials
берутся из окружения SSH-инструмента. Инстанс после привязки не меняется. Повтор `launch`
при активном job выполняет `--attach`; `resume` требует завершённого `pull` предыдущего
шага (после которого локальный и удалённый `config["output"]` идентичны по SHA-256).

`status` читает ограниченный живой срез по SSH. `pull` дожидается завершения активного job,
проверяет `execution`, `post-sync` и доставку архивного пакета (если включена), запускает научный `audit` напрямую
на `config["output"]` и по умолчанию сохраняет инстанс; `--delete-instance` удаляет
только привязанный ID после прохождения обязательных секций аудита.
