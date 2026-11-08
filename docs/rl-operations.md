# Работа с RL-запусками

Сверено с CLI и реализацией 8 ноября 2026 года. Команды выполняются из корня
репозитория в окружении опыта. `RUN`, `CONFIG`, `CHECKPOINT`, `ROLLOUT` и
`OUT` ниже — подставляемые пути. Условия обучения, parent и критерии выбора
модели ищите в [журнале](experiment-ledger.md), устройство паспортов — в
[архитектуре](rl-architecture.md#паспорта-исходники-и-resume).

## Анализ сохранённого запуска

```powershell
python tools/ctl_rl.py audit --run RUN --out OUT
python tools/ctl_rl.py compare --before BEFORE --after AFTER --expected-n 128 --out OUT --clusters 16 --samples 10000 --seed 20261103
python tools/ctl_rl.py snapshot --run RUN --out OUT --block-size 5 --last 20
python tools/ctl_rl.py performance --run RUN --last 10
```

`audit` проверяет замороженный паспорт/архив, транзакционный PPO ledger,
CPU checkpoint/Adam/RNG, полноту сохранённых оценок и W/D/L, парное сравнение.
Он не сравнивает текущее рабочее дерево. Для старого паспорта без hash
resolved config выводит `config_hash_verified: false`; недоступные проверки
не объявляются пройденными. Deployment passport отмечается отдельно от
паспорта runner. Проверка архива не доказывает корректность всех результатов.

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
поддерживает update boundary; target-scale imitation — полностью сохранённую
эпоху. Неподдерживаемые сочетания отклоняются. Плановая оценка на границе
заканчивается до паузы; дополнительная final evaluation ради паузы не запускается.
Старый запущенный процесс не приобретает новые hooks от правки файлов.

PPO resume удаляет запрос и продолжает сохранённые model/Adam/RNG/счётчики.
У imitation нет общего безопасного pause-resume; специализированный старый
input-fix restart нельзя считать таким продолжением. У target-scale PPO
есть также собственные `pause`/`resume` в `tools/run_ppo_scale_up.py`.
Поддержку нужно выбирать по runner, а не только по наличию файла запроса.

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
предпочтение отдаётся меньшему числу workers/сред и меньшему ожиданию/reply.
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

`tools/remote_rl.py` работает с уже существующим инстансом. Аренда и выбор
машины выполняются отдельно; по умолчанию проекта при создании Vast-инстанса
готовятся два кандидата (`--parallel 2`).

```powershell
python tools/remote_rl.py package --config configs/rl-ppo-target-bc-t05.json --out DEPLOYMENT --extra-input CASE_CARRIER.pt
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode tune --instance INSTANCE_JSON
python tools/remote_rl.py pull --deployment DEPLOYMENT
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode run
python tools/remote_rl.py status --deployment DEPLOYMENT
python tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode resume
```

| Kind | run | tune | resume |
|---|---|---|---|
| `rl01.ppo_after_target_bc` | Да | Да | Да |
| `rl01.ppo_target_scale` | Да | Нет | Да |
| `rl01.ppo_scale_up`, включая continuation | Да | Нет | Нет |

BC не входит в этот интерфейс. Каталог package должен быть пустым.
Для after-target-BC tune требуется ровно один явно добавленный `.pt` case
carrier с нужными фиксированными случаями; расширение файла не доказывает
его семантическую пригодность. Tune ограничен 600 секундами; перед run нужен
подтверждённый comparable tune для точного package/config и его pull.

### Состав пакета и недостающие входы

Упаковка не обходит транзитивно произвольные manifests. Она захватывает
parent checkpoint/manifest, configs и открытые карты из выбранного manifest,
но не гарантирует карты другого corpus manifest или исторический rollout,
который runner читает отдельно. Для target-scale prepare нужны открытые карты
`artifacts/rl/corpus/manifest.json`; target-scale и scale-up math check читают
`artifacts/rl/ppo-controlled-updates/rollouts/first.pt`.
До исправления упаковщика эти зависимости передаются явно. Для большого
списка карт на Windows используйте тот же Python API внутри одного процесса:
сотни `--extra-input` могут превысить предел длины командной строки.

```powershell
@'
import json
from pathlib import Path
from civ5rl.deploy.operations import package
corpus = json.loads(Path('artifacts/rl/corpus/manifest.json').read_text(encoding='utf-8'))
extras = ['artifacts/rl/corpus/manifest.json', 'artifacts/rl/ppo-controlled-updates/rollouts/first.pt']
extras += [m['path'] for m in corpus['maps'] if m['split'] in ('train', 'validation')]
print(json.dumps(package('configs/rl-ppo-target-scale.json', 'DEPLOYMENT', extra_inputs=extras), indent=2))
'@ | python -
```

Передать только manifest недостаточно. Дополнительные пути должны находиться
в репозитории. Код сборщика входов исключает test splits при обходе manifest,
но `source.zip` отдельно включает tracked/untracked non-ignored файлы;
`extra_inputs` тоже может включить test. Поэтому общая гарантия «закрытые карты
не попадут в пакет» из фильтра inputs.zip не следует: состав архивов нужно
сверять до запуска. Эти ограничения описывают текущий код; документационная
ревизия его не исправляет.

### Жизненный цикл deployment

`INSTANCE_JSON` задаёт instance_id/id, host/SSH_host, port/SSH_port; credentials
берутся из окружения SSH-инструмента, а не встраиваются в протокол. Инстанс
после привязки не меняется. Fingerprint фиксирует sources/inputs, размеры,
SHA-256 и членов архивов; Git bundle сохраняет base commit.
Рабочие пути выводятся из fingerprint под `/workspace/civ5-runs/`.

Исполнение использует durable runner установленного remote-python-ssh skill,
отдельный payload mirror и verified post-sync. Проект, venv и scientific output
находятся вне этого mirror. Повтор launch присоединяется к связанному job;
неопределённый старт восстанавливается по durable registry без слепого повтора.
`resume` требует прежние sources/inputs и сохранённые model/Adam/RNG/budget
на том же инстансе; локальные изменения поверх run не накладываются.

`status` читает ограниченный срез. `pull` ждёт завершения активного job,
проверяет transport/post-sync, operation token, ZIP safety и hashes перед
извлечением. Scientific audit записывается отдельно от целостности транспорта.
Повторный pull принимает идентичные файлы и отклоняет конфликтующие.
Инстанс сохраняется по умолчанию; `pull --delete-instance` удаляет только
привязанный ID после успешных обязательных проверок. Ошибки выгрузки/проверки
не превращаются в успешную передачу или разрешение удалить машину.
