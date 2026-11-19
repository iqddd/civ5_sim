# История оптимизаций

Оптимизации исполнения, замеры производительности и отклонённые варианты.
Обучение, корпуса и сравнения политик — в [журнале экспериментов](experiment-ledger.md).

## Whole-turn fit

- CPU-замер на 7000 контрольных опорах и 64 позициях шага, медиана трёх
  повторов. Построение маски вне учебных продолжений занимало 10,25 с на
  проверку и перечитывало 595 историй. Подготовка маски выполняется один раз
  за сессию (11,04 с), повторный поиск готовых масок занимает 0,14 мс.
  После подготовки предпочтений упаковка батча занимает 8,3 мс на шаг вместо
  0,89 с на попытку. Повторы используют готовый батч. Байты батча и число опор
  вне учебных продолжений совпали. Замер исключает вычисления модели и перенос
  данных на GPU. [Замер](../artifacts/preparations/c2p-turn-fit-cached-inputs-timing/comparison.json).
- RTX 4060 Ti, FP32, штатный шаг 64 позиций / 660 опор / 7000 контрольных опор
  (`--history-microbatch 2`): один граф на позицию через матрицу выбора без
  отдельного прохода коэффициентов, сжатое префиксное дерево для `no_grad`,
  батчи `no_grad` GRU по 64 без склейки лишних `x` и перенос скрытых состояний
  следующих опор из `control_kl`. Прогретый цикл: 23,65 с шаг + 14,79 с
  полный KL + 0,11 с сохранения = **38,55 с** (прогрев 57,57 с) против 67,06 с
  (прогрев 122,66 с) до переделки; средняя NVML Core Load 73,1%, мощность
  72,2% TDP. В `precheck` метрики и веса `value` совпали, `indexing_backward`
  отсутствует; 4 целевых теста пройдены.
  [Замер](../artifacts/preparations/c2p-turn-fit-prefix-tree-bench/benchmark.json),
  [NVML](../artifacts/preparations/c2p-turn-fit-prefix-tree-bench/resources-summary.json),
  [smoke](../artifacts/preparations/c2p-turn-fit-prefix-tree-bench/precheck.json).
- RTX 4090 / i9-14900KF (Vast.ai, instance 61390071), полный fit
  `c2p-turn-comparison-fit-v1` (FP32, `--history-microbatch 2`, 16 принятых
  шагов / 0 отклонений): **404,3 с** суммарно с подготовкой и сохранениями.
  Шаг 1 (с компиляцией): 37,71 с шаг + 5,01 с полный KL; прогретые шаги 2–16:
  6,34–7,03 с шаг + 4,55–4,97 с полный KL (~11,1 с цикл). В фазе `training`
  (202 выборки NVML): средняя GPU Core Load **80,9%**, мощность **364,3 Вт
  (81,0% TDP)**, пик VRAM **6,56 GiB**.
  [Training](../artifacts/rl/c2p-turn-comparison-fit-v1/training.json),
  [Steps](../artifacts/rl/c2p-turn-comparison-fit-v1/steps.jsonl),
  [NVML](../artifacts/rl/c2p-turn-comparison-fit-v1/telemetry/resources.jsonl).

## Action-value head smoke

- RTX4060Ti, FP32, batch256forks/512actions: warmed head step **2.08ms** (3warmup/5measured); excludes saving/preparation. Real cached u171 vectors reproduced141action logits exactly. Synthetic controller interrupted/resumed with identical weights/Adam; no full fit or matches. Short smoke phase is below the utilization-policy duration threshold. [Smoke and kernel trace](../artifacts/preparations/c2p-action-value-smoke-verified/precheck.json).

- Action-value execution: epoch-boundary checkpoints, one permutation upload per epoch, device loss accumulation, resident training/selection splits. Cache-hit preparation loads no policy and runs no head diagnostics; immutable source packets are verified and retained without recompression. Bounded smoke reproduces original update weights bitwise and recovers an interrupted epoch exactly; no full fit was run. [Evidence](../artifacts/preparations/c2p-action-value-optimized-smoke-final/precheck.json).

- Full action-value execution on RTX4060Ti: preparation108.09s with1028/1028feature-cache hits and zero policy loads; 100epochs/1000updates including selection/checkpointing completed in4.37s. First-epoch forecast29.8s was conservative because of startup. The iterative phase is shorter than the utilization-policy threshold. [Timing](../artifacts/rl/c2p-action-value-v1/result.json).

- Whole-turn collection smoke: central CUDA FP32 inference, task-local sampling RNG, switch back to argmax without resetting memory.16discardable2-round continuations plus order/prefix-cache checks completed in21.04s including setup; source prefix prefilled once and reused. NVML and Kineto recorded; this short phase is below the utilization-policy threshold. This is not a forecast for60-round collection. [Smoke](../artifacts/preparations/c2p-turn-comparison-final-smoke/precheck.json).

## Отложенные задачи

- [x] Обработать временный Windows `PermissionError` при атомарной публикации
  heartbeat бюджета, чтобы фоновая запись продолжалась после отказа.
  [Наблюдение](../artifacts/rl/c2p-rollout-comparison-v2-heads/console.log):
  основной deadline работал, итоговое время сохранено корректно.

- [ ] Включить `prefix_states` по умолчанию в сравнительном сборе, сохранив явное
  отключение и запись выбранного режима в runtime. Проверить итоговый корпус
  через финализацию, импорт и восстановление учебных историй.
- [ ] Перенести отдельный inference-процесс из стенда в штатный execution-путь.
  Проверить коротким сравнительным smoke исходы, изоляцию состояний и завершение
  процессов; после проверки выбрать стандартный режим.
- [ ] Завершить разбор недозагрузки CPU/GPU и масштабирования воркеров после
  внесённых оптимизаций: разделить симулятор/CP, кодирование, IPC и ожидания
  inference. Сопоставлять скорость и загрузку в одном рабочем окне, отдельно
  от прогрева и хвоста. Часть затрат хеширования и повторного префикса уже
  устранена; подтверждённые замеры приведены ниже.

## 2 ноября 2026 — поиск и хранение CP

- **Варианты поиска:** на 49 входах distant26 base — 17,355 с,
  distance-view — 7,461 с, area-strict-bundle — 0,442 с. Часть вариантов меняла
  ответ; эти замеры не устанавливают силу AI.
  [Результаты](../artifacts/combat-ai/experiments-v1/report.md),
  [исторический протокол](archive/combat-ai/combat-ai-experiments.md).
- **Перенос хранения:** один прогрев и пять замеров, исторический ABI 5,
  нулевая история урона. Изменения эвристик смешаны с хранением; RSS измерялся
  как high-water процесса.
  [Результаты](../artifacts/combat-ai/storage-transfer-20261102/report.md),
  [карта переноса](combat-ai/combat-ai-cp-transfer.md).

## PPO C2 h60 GAE — память и overhead

Разбор накопления VRAM, предварительной проверки памяти и затрат времени
с локальной проверкой исправлений. Историческое обучение не повторялось;
продление бюджета и смены исходников записаны в сохранённом запуске.
[Разбор](../artifacts/rl/ppo-c2-h60-gae/derived/memory-overhead-review/report.md),
[ревизии исполнения](../artifacts/rl/ppo-c2-h60-gae/source-revisions/).

## 12 ноября 2026 — C2 attention/backward

На RTX 4060 Ti masked reduction вместо one-hot/GEMM уменьшила PPO backward
с 323 до 294 мс; полные updates в свежих процессах — с 2,68 до 2,49 с
(по два замера). SDPA/Flex/BF16 не приняты по скорости; автоматический compact-
прототип не прошёл replay. Специализированное Triton-ядро ускоряло backward,
но преимущества полного update не подтвердило. Это короткая диагностика,
без оценки силы политики или прогноза скорости RTX 4090.
[Замеры и проверки](../artifacts/benchmarks/c2-attention-20261112/report.md).

## 15 ноября 2026 — сбор и обучение

### Legacy/end_turn и холодный генератор

RTX 5090, checkout `d247575` и текущий на момент замера, update 171/update 20:
на 80/24/1 получено 235,27/234,47 решений/с. Замедление от `end_turn`
не воспроизвелось. На одинаковых 18 258 решениях холодный factory занял
27,37 с, прогретый — 12,27 с; команды совпали. На 42 visibility cache misses
пришлось 13,26 с. Холодный генератор и больший rollout поддержали гипотезу
о падении скорости с 340 до 240 решений/с; прогрева одной GPU недостаточно.
[Полные циклы](../artifacts/benchmarks/c2p-live-cycle-5090-20261115/report.md),
[контроль причины](../artifacts/benchmarks/c2p-slowdown-cause-20261115/report.md).

### Rollout comparison: профиль CPU/GPU и компиляция

Локально, i5-11600K / RTX 4060 Ti, сохранённые training cases:

| Фаза | Время | GPU Core / Controller |
|---|---:|---:|
| 8 исходных боёв | 17,44 с | 30,3% / 10,8% |
| 16 продолжений | 11,38 с | 29,0% / 23,7% |
| Учебный шаг, batch 32+64, BPTT 128 | 29,50 с | 74,3% / 69,1% |

Обучение измерялось после одного прогрева, на трёх отбрасываемых шагах;
full KL/references в время шага не входили. Мощность — около 90 Вт.
В короткой трассе видны затраты eager-запусков и материализаций;
stream-sync/HtoD не доминировали.
[Профиль](../artifacts/benchmarks/rollout-comparison-cpu-gpu-20261115/report.md).

`torch.compile` для spatial/heads в режиме default, без CUDA Graph pools,
сократил шаг до 7,13–7,69 с. BF16 gradient L2δ — 1,295%, distribution KL —
4,51×10⁻⁵; FP32 logits δ — 9,54×10⁻⁶, сохранённый replay воспроизведён.
Короткий замер NVML: Core 61%, Controller 45%. Полный опыт не выполнялся.
[Проверка компиляции](../artifacts/benchmarks/comparison-compile-20261115/report.md).

## 16 ноября 2026 — масштабирование сравнительного сбора

### RTX 4090 / EPYC 7742

Для v2-сбора W80 выбирался по 10,47 продолжения/с. Исходные 26,7 занятых
логических ядер при квоте 61,44 потребовали отдельного разбора ожиданий
воркеров, inference-сервиса, IPC и симулятора/CP.
[Подбор W64/W80](../artifacts/benchmarks/comparison-collection-4090-w64-w80-20261116/measurement/summary.json).

- **Snapshot hashing (`84d56f0`):** на W80, одинаковых 128 развилках и
  1224 продолжениях, ABBA дал 129,35/126,79 с до и 107,40/110,41 с после
  устранения копирования неизменяемых карты/правил. Среднее время −14,96%,
  пропускная способность +17,60%. В каждом проходе Δ₁ совпала для 128/128
  развилок, replay и входные hashes проверены.
  [Парные замеры](../artifacts/benchmarks/collection-paired-results.json),
  [профили](../artifacts/benchmarks/collection-profile3/diagnostics/),
  [тайминги воркеров](../artifacts/benchmarks/collection-scaling-optimized/worker-timing/).
- **Hidden в точке ветвления (`da9b6e1`):** W64, один отдельный inference-
  процесс, те же 128 развилок/1224 продолжения. Время 127,30 → 89,03 с
  (−30,1%; throughput +43,0%), рабочая скорость 9,69 → 13,88 cont/s.
  Устранено 135 312 observe RPC; включены 32 prefill исторических исходных
  боёв. Δ₁ совпала для 128/128 развилок, replay и входы проверены.
  Проверка свежего сбора: 4 развилки/48 продолжений, 4352 пропущенных RPC,
  без prefill. Проверены изоляция hidden и публичный encoder; прошли 19 тестов.
  [Baseline](../artifacts/benchmarks/branch-prefix-baseline64/summary.json),
  [hidden на reset](../artifacts/benchmarks/branch-prefix-enabled64/summary.json),
  [свежий сбор](../artifacts/benchmarks/branch-prefix-live-smoke/collection.json).

Кэш по fingerprint каждого запроса после IPC и альтернативная телеметрия
отклонены после замедления. Это отдельные варианты, а не отрицательный
результат восстановления hidden в точке ветвления.

### RTX 4090 / Ryzen 9 7950X3D

Квота 30,72 CPU, W32, один inference-процесс, `prefix_states` включён.
На тех же 128 развилках/1224 продолжениях два baseline и два итоговых замера:
71,49/71,35 → 57,91/57,72 с. Среднее время −19,05%, throughput +23,53%;
рабочая скорость 17,80 → 21,99 cont/s. Изменения: повторное использование
статической части хеширования, устранение дублирующего hash префикса и
копирования снимков при replay; постоянный selector и чтение shared views.
Канонические SHA сохранены, все 1224 исхода совпали в парном повторе;
Δ₁ 128/128, replay и входы проверены.
[Baseline A](../artifacts/benchmarks/newhost-service-baseline32-a/summary.json),
[baseline B](../artifacts/benchmarks/newhost-service-baseline32-b/summary.json),
[итог A](../artifacts/benchmarks/newhost-service-digest32-a/summary.json),
[итог B](../artifacts/benchmarks/newhost-service-digest32-b/summary.json),
[профили](../artifacts/benchmarks/newhost-service-profile/diagnostics/).

Отдельные варианты: selector — 71,33 с, shared views — 70,21 с;
pinned-буферы — 70,61 с, вариант убран. Два сервиса дали 57,37 с —
разницу менее 1%; Core Load вырос до 32,2% с 15,2%, RSS — до 33,9 с 20,0 GiB.
Прошли десять целевых проверок; отдельный CP-oracle тест недоступен из-за
отсутствия тестового экспорта в перенесённой библиотеке.
[Два сервиса](../artifacts/benchmarks/newhost-service-digest32-two-services/summary.json).

## 17 ноября 2026 — число воркеров и полный прогрев

RTX 4090 / Ryzen 9 7950X3D, один inference-процесс, `max_batch=W`,
`prefix_states`. Каждый режим выполнял два одинаковых прохода по 128 развилкам
и 1224 продолжениям; измерялся второй. До замера все W процессов были запущены.

| W | Рабочая скорость, cont/s | Полное время с хвостом, с |
|---:|---:|---:|
| 24 | 20,10 | 60,92 |
| 32 | 22,75 | 56,10 |
| 40 | 23,71 | 54,00 |
| 64 | 24,06 | 51,36 |
| 80 | 23,53 | 51,53 |

W40 выбран как режим в пределах 5% лучшего прогноза. Все 1224 исхода совпали
во всех режимах, replay и входы проверены. Первоначальный короткий прогрев
включал создание части процессов W40/W64/W80 в измерение; в стенд добавлен
`--warmup-measured-work`. Обучение не запускалось.
[Полностью прогретое сравнение](../artifacts/benchmarks/newhost-worker-grid-warm-24-32-40-64-80/summary.json),
[runtime](../artifacts/benchmarks/newhost-worker-grid-warm-24-32-40-64-80/recommended-runtime.json).

### Резидентные GRU hidden

RTX 4090 / Ryzen 9 7950X3D, W40, один inference-процесс. Постоянные слоты
hidden и буфер батча обслуживаются через foreach-копии. На 40 состояниях:
249 → 42 мкс на итерацию (три прогрева, пять замеров); в Kineto за три
повтора — 246 → 9 GPU-событий. Значения hidden побитово совпали.
Полный сбор после отдельного полного прогрева: 53,79 → 53,59 с на тех же
128 развилках/1224 продолжениях. Разница 0,37% не доказывает общего ускорения.
Все 1224 исхода совпали с контролем; десять целевых проверок прошли, включая
reset, prefill, повторное использование слотов и отсутствие роста живой VRAM.
[Baseline](../artifacts/benchmarks/resident-hidden-baseline40/summary.json),
[резидентные состояния](../artifacts/benchmarks/resident-hidden-enabled40/summary.json),
[изолированный замер и трассы](../artifacts/benchmarks/resident-hidden-copy-trace/summary.json).

## 17 ноября 2026 — FP32-кэш для обучения голов

RTX 4060 Ti, 330 пар + 660 учебных опор, контроль на 12096 позициях: после
одного прогрева шаг голов занял 0,235 с против 18,23 с полнокорпусного
обучения; полный KL с диагностикой — 2,30 с против 22,59 с. Замороженные
признаки вычислены в FP32, сравнение скорости относится к разным режимам
обучения (в исходном пространственная часть mixed). Холодная подготовка —
336,1 с, кэш — 1,56 GiB; прогноз 10 шагов — 367,7 с без финальной упаковки.
Короткая трасса шага: H2D 46,3 мс, GEMM 19,6 мс; рабочее окно короче
10 минут, средняя NVML Core Load 15,1%, Memory Controller 3,4%, мощность
24,7 Вт. Кэшированный и обычный FP32-forward на трёх позициях: максимальная
разница logits 1,06e-5, argmax совпал.
[Замер](../artifacts/preparations/c2p-heads-local-timing/summary.json),
[ядра](../artifacts/preparations/c2p-heads-local-timing/top-kernels.json),
[трасса](../artifacts/preparations/c2p-heads-local-timing/heads-trace.json).
