# Работа с RL-запусками

Сверено с CLI и реализацией 11 ноября 2026 года. Команды выполняются из корня
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

### Подготовленный профиль продолжения C2 от update 150

`configs/rl-ppo-c2-h60-gae-continuation.json` — отдельный новый опыт, а не resume
исторического каталога. Родительские входы извлечены из
`artifacts/rl/ppo-c2-h60-gae/ppo-c2-h60-gae.7z` в
`artifacts/inputs/ppo-c2-h60-gae/`: `checkpoints/update-0150.pt`, `manifest.json`,
`state.json`, `run.json`, `derived/review-20261111/analysis.json`.
Hashes и проверка подготовки: [отчёт](../artifacts/preparations/ppo-c2-h60-gae-continuation/report.md).
Переупаковка родительских результатов, извлечение evaluation или test не нужны.
`stage_inputs` автоматически включает родителя, manifest, конечный state,
run record, исторический прогноз и только открытые карты.

Подготовка кода не запускает научный каталог или полный Development. Будущие
команды ниже исполняются только после отдельного согласования запуска;
package требует чистого дерева и отправленного commit, инстанс выбирается
отдельно по согласованному пресету. Для tune нужен существующий совместимый
rollout-носитель (`CARRIER`); он только задаёт фиксированные открытые случаи.

```powershell
$cfg = 'configs/rl-ppo-c2-h60-gae-continuation.json'
$dep = 'artifacts/deployments/ppo-c2-h60-gae-continuation'
.venv/Scripts/python.exe tools/remote_rl.py package --config $cfg --out $dep --instance .tmp/instance.json --extra-input CARRIER
.venv/Scripts/python.exe tools/remote_rl.py launch --deployment $dep --mode tune
.venv/Scripts/python.exe tools/remote_rl.py pull --deployment $dep --keep-instance
.venv/Scripts/python.exe tools/remote_rl.py launch --deployment $dep --mode run --evaluation-device cuda --evaluation-workers 8 --evaluation-max-batch 32 --evaluation-wait-seconds 0.001
.venv/Scripts/python.exe tools/ctl_rl.py progress --deployment $dep --watch --interval 900
```

Если процесс оборвался, сначала `pull --keep-instance`, затем:

```powershell
.venv/Scripts/python.exe tools/remote_rl.py launch --deployment $dep --mode resume --evaluation-device cuda --evaluation-workers 8 --evaluation-max-batch 32 --evaluation-wait-seconds 0.001
```

Runtime оценок закреплён также конфигом: отсутствие флагов не включает auto,
а отличающиеся значения отклоняются. Перед первым сбором выполняется новая
CUDA/FP32 нулевая Development-512, задающая W0 и постоянный порог W0+11.
Отдельный benchmark не нужен: эта оценка даёт длительность полного Development.
Исторические поздние окна дают 5,8–7,2 активного часа на 8 Mi; оценки и возможные
сбои добавляют wall time, на новой машине скорость не гарантируется.
Первые 2–3 принятых шага проверяются короткими проверками, затем — раз в 15 минут.

После окончательных Development/Additional выгрузить и проверить артефакты:

```powershell
.venv/Scripts/python.exe tools/remote_rl.py pull --deployment $dep --keep-instance
.venv/Scripts/python.exe tools/ctl_rl.py audit --run artifacts/rl/ppo-c2-h60-gae-continuation --out artifacts/rl/ppo-c2-h60-gae-continuation/derived/audit
```

После подтверждения синхронизации и обязательных секций аудита удалить привязанный
инстанс штатной командой Vast.ai из раздела ниже. Повтор `resume` завершённого
обучения не начинает новые решения; завершённый Additional берётся из сохранённого
результата. Контрольные проверки подготовки выполняются в отдельном временном
каталоге и не являются ни первым rollout опыта, ни нулевой оценкой.

`tools/remote_rl.py` работает с уже существующим инстансом поверх `remote-python-ssh`.
Аренда и выбор машины выполняются отдельно; по умолчанию проекта при создании
Vast-инстанса готовятся два кандидата (`--parallel 2`). При запуске удалённого
опыта не нужно перечитывать исходники `civ5rl/deploy/` или `tools/tune_rl_collection.py` —
выполняется сразу штатная последовательность команд ниже.

На Windows инструменты репозитория (`tools/remote_rl.py`, `tools/ctl_rl.py`)
запускаются через `.venv\Scripts\python.exe` (где установлены `lupa`, `torch` и
зависимости симулятора), а менеджер инстансов `vastai_instance.py` — через
системный `python` (где установлен `keyring` для Vast API ключа; поэтому при
завершении удобнее вызвать `pull --keep-instance` через `.venv\Scripts\python.exe`,
а затем явно удалить инстанс через `vastai_instance.py delete <ID>`). Для
`rl01.ppo_after_target_bc` при упаковке с `--mode tune` в `--extra-input`
передаётся готовый роллаут-носитель расстановок (например,
`artifacts/rl/ppo-target-bc-t05/rollouts/rollout-0001.pt`).

```powershell
git push origin HEAD
python C:\Users\Roman\.codex\skills\vastai-instance-manager\scripts\vastai_instance.py create --profile 4090 --parallel 2 --save-instance .tmp/instance.json
.venv\Scripts\python.exe tools/remote_rl.py package --config configs/rl-ppo-c2-h60-gae.json --out artifacts/deployments/ppo-c2-h60-gae --instance .tmp/instance.json --extra-input artifacts/rl/ppo-target-bc-t05/rollouts/rollout-0001.pt
.venv\Scripts\python.exe tools/remote_rl.py launch --deployment artifacts/deployments/ppo-c2-h60-gae --mode tune
.venv\Scripts\python.exe tools/remote_rl.py pull --deployment artifacts/deployments/ppo-c2-h60-gae --keep-instance
.venv\Scripts\python.exe tools/remote_rl.py launch --deployment artifacts/deployments/ppo-c2-h60-gae --mode run
.venv\Scripts\python.exe tools/remote_rl.py status --deployment artifacts/deployments/ppo-c2-h60-gae
.venv\Scripts\python.exe tools/remote_rl.py pull --deployment artifacts/deployments/ppo-c2-h60-gae --keep-instance
.venv\Scripts\python.exe tools/remote_rl.py launch --deployment artifacts/deployments/ppo-c2-h60-gae --mode resume
python C:\Users\Roman\.codex\skills\vastai-instance-manager\scripts\vastai_instance.py delete INSTANCE_ID
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

### Подготовленная ветка gamma=1

`configs/rl-ppo-c2-h60-gae-gamma1.json` ответвляется от того же update 150,
сохраняет Adam общей сети/actor и адаптирует value head под недисконтированный
исход. Это отдельный опыт: 4 Mi новых решений или пять активных часов,
Development 0/1/2/3/4 Mi и gate W0+11 на 2 Mi. Подготовка critic входит в бюджет.
Исторический темп 323–400 dec/s даёт 2,91–3,61 часа на 4 Mi; полный недавний
запуск с компиляцией дал около 282 dec/s (около 4,13 часа при экстраполяции).
Дополнительно учесть до 300 секунд подготовки и refresh; скорость новой машины
не гарантируется. Отдельный benchmark и полный Development при подготовке
не запускаются. Исторический запуск gamma1 завершён; результат и выбранный update 147 см. в ledger.

Локальный запуск после отдельного согласования:

```powershell
$cfg = 'configs/rl-ppo-c2-h60-gae-gamma1.json'
$run = 'artifacts/rl/ppo-c2-h60-gae-gamma1'
.venv/Scripts/python.exe tools/run_ppo.py run --config $cfg --runtime-config RUNTIME
.venv/Scripts/python.exe tools/ctl_rl.py progress --run $run --watch --interval 900
.venv/Scripts/python.exe tools/ctl_rl.py request-pause --run $run --after-update UPDATE
.venv/Scripts/python.exe tools/run_ppo.py resume --config $cfg --runtime-config RUNTIME
.venv/Scripts/python.exe tools/ctl_rl.py audit --run $run --out artifacts/rl/ppo-c2-h60-gae-gamma1/derived/audit
```

CUDA/FP32 runtime оценок закреплён самим профилем. Для удалённого исполнения
использовать ту же последовательность package/tune/pull/run/resume выше, заменив
config и deployment на gamma1. Сначала выгрузить завершившийся job, затем resume.
При сбое подготовки сохранять вместе pending checkpoint, первый rollout,
`critic-adaptation.pt` и `critic-features.pt`; все они входят в обычный mirror.
После `critic-ready` resume использует полный checkpoint и не повторяет подготовку.
Для окончательного завершения — `run`/`resume` с уже установленным stop_reason,
затем pull, аудит и удаление только отдельно разрешённого инстанса.

### Продолжение gamma=1 от update 147

Этот опыт завершён по gate на 2 Mi; повторная аренда для него не требуется.
Результаты и проверенные артефакты — в [ledger](experiment-ledger.md).

`configs/rl-ppo-c2-h60-gae-gamma1-continuation.json` — новый опыт, не resume
завершённого gamma1. Родительские checkpoint/manifest/final state/run record
зафиксированы в `artifacts/inputs/ppo-c2-h60-gae-gamma1/`. Сохраняются actor/shared
Adam 25171 и value Adam 12615, critic не готовится повторно. Первый episode ID
52399; новые namespaces и seed 17. Бюджет 8 Mi или 36000 active seconds;
Development 0/1/2/4/6/8 Mi и gate W0+11 на 2 Mi. Нулевую оценку пересчитать
на арендованной машине; при выборе нового кандидата обе модели получают
один paired Additional-256. Test закрыт.

Использовать последовательность package/tune/pull/run выше с новым config и
отдельным deployment `artifacts/deployments/ppo-c2-h60-gae-gamma1-continuation-5090`.
Для tune передать `--extra-input artifacts/rl/ppo-c2-h60-gae-gamma1/rollouts/latest.pt`.
Родительский rollout служит только носителем открытых случаев подбора.
Плановые перезапуски отключены; `--max-updates` и pause.request при запуске
не задаются. Телеметрия cuda-memory включена; после 2–3 подтверждённых шагов
проверки раз в 15 минут. После финальных оценок — pull, аудит, удаление инстанса.
Прогноз по родительским поздним окнам 7,9–9,4 активного часа на 8 Mi, без
гарантии новой машины. Оценки не расходуют active budget.

### CPU-фильтр будущей аренды

Этот фильтр применяется только к задачам, которым требуется CPU-зависимая
фаза, например сбор боёв симулятором для PPO. Для обучения BC без такой фазы
он не нужен. Применимость определяется составом планируемых фаз, а не только
названием runner. Для задач с CPU-зависимой фазой, помимо согласованного
GPU-профиля, цены, географии и сетевых требований, кандидат должен удовлетворять:

```text
cpu_score = cpu_cores_effective × (100 × PassMark Single Thread / 2067)
cpu_score >= 1250
```

`cpu_cores_effective` — доступные логические CPU из конкретного предложения
Vast.ai, включая дробную долю. Не подставлять полное число потоков процессора
или физические ядра. 2067 — фиксированная база EPYC 7702 (100%); в формуле
используются процентные пункты, например 229, а не 2,29. Для решения о допуске
не округлять отношение. Это дополнительное правило отбора проекта; сама команда
менеджера `search` этот вычисляемый фильтр не применяет. Проектный вызов
`python tools/remote_rl.py rent --cpu-score-min 1250` принимает остальные флаги
штатного `create`, включая `--profile`, и накладывает CPU-допуск на каждый
свежий отбор. Неизвестная доступная по цене модель останавливает выбор с
перечнем для PassMark lookup, а не молча исключается. Без `--cpu-score-min`
дополнительного CPU-фильтра нет (например, для BC). До аренды отсеивать
неподходящие предложения и не выбирать их повторно по одной лишь низкой цене.

Исходные Single Thread ниже предоставлены пользователем. Проценты округлены
только для отображения. Если точной модели нет, найти её PassMark Single Thread
на CPU Benchmark, дописать значение и ссылку на источник сюда, затем вычислить
допуск. Не заменять неизвестную модель похожим названием или Multi Thread score.

| CPU | PassMark Single Thread | К EPYC 7702 |
|---|---:|---:|
| Intel Core Ultra 9 285K | 5087 | 246% |
| AMD Ryzen 9 9950X | 4726 | 229% |
| Intel Core i9-13900K | 4594 | 222% |
| AMD Threadripper PRO 9975WX | 4428 | 214% |
| AMD Threadripper PRO 7975WX | 3989 | 193% |
| AMD EPYC 9554 | 2774 | 134% |
| AMD EPYC 9124 | 2613 | 126% |
| AMD EPYC 7702 | 2067 | 100% |
| AMD EPYC 7302 | 2024 | 98% |
| AMD EPYC 7532 | 2015 | 97% |
| AMD EPYC 7313P | 2576 | 125% |
| AMD EPYC 7542 | 1957 | 95% |
| AMD EPYC 9135 | 3228 | 156% |
| AMD EPYC 9355 | 3747 | 181% |
| Intel Core i9-14900KF | 4688 | 227% |
| AMD Ryzen Threadripper 3960X | 2681 | 130% |
| Intel Core Ultra 7 270K Plus | 5073 | 245% |
| AMD Ryzen 9 9900X | 4670 | 226% |
| AMD Ryzen 9 7900X3D | 4117 | 199% |
| AMD EPYC 7713 | 2648 | 128% |
| AMD Ryzen 7 9800X3D | 4420 | 214% |
| AMD EPYC 9115 | 3360 | 163% |
| AMD EPYC 8224P | 2341 | 113% |
| Intel Core i5-13600K | 4110 | 199% |
| Intel Core i7-13700K | 4324 | 209% |
| AMD EPYC 9254 | 2998 | 145% |
| AMD Ryzen 7 3700X | 2654 | 128% |
| AMD Ryzen 9 9950X3D | 4737 | 229% |
| AMD Ryzen 9 5950X | 3476 | 168% |
| AMD Ryzen 9 7940HX with Radeon Graphics | 3925 | 190% |

Threadripper 3960X, Ultra 7 270K Plus, Ryzen 9900X/7900X3D и EPYC 7713 дополнены по [PassMark Single Thread](https://www.cpubenchmark.net/singleThread.html)
12 ноября 2026 года; исходные пользовательские значения выше не пересмотрены.
Ryzen 9800X3D — [4420](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+7+9800X3D),
EPYC 9115 — [3360](https://www.cpubenchmark.net/cpu.php?cpu=AMD+EPYC+9115),
EPYC 8224P — [2341](https://www.cpubenchmark.net/cpu.php?cpu=AMD+EPYC+8224P)
добавлены по PassMark 13 ноября 2026 года. Ryzen 9950X3D: [4737](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+9+9950X3D&id=6549); не путать с X3D2. Ryzen 7940HX: [3925](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+9+7940HX). Ryzen 5950X: [3476](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+9+5950X&id=3862).
При свежем отборе также проверены [Intel Core i5-13600K: 4110](https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i5-13600K), [Intel Core i7-13700K: 4324](https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i7-13700K), [AMD EPYC 9254: 2998](https://www.cpubenchmark.net/cpu.php?cpu=AMD+EPYC+9254), [AMD Ryzen 7 3700X: 2654](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+7+3700X).

Например, EPYC 7313P с долей 6,4 получает 797,6 и не проходит, а Ryzen 9950X
с долей 32 получает 7316,5 и проходит. После аренды фактическая quota/cpuset
фиксируется отдельно: рекламная доля и этот эвристический балл не гарантируют
скорость симулятора или CPU-инференса. Новый фильтр не даёт разрешения на аренду.

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
