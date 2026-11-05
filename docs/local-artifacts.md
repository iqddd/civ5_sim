# Локальные артефакты и восстановленный минимум

`artifacts/` и `build/` полностью исключены из Git. Ранее отслеживавшиеся
артефакты удалены только из индекса; существующие локальные файлы сохраняются.
Исходники, конфигурации, протоколы и тесты остаются в репозитории.

Архивы `build/build-history.7z` и
`artifacts/rl/development-history-2.7z` сохраняются без изменений.

## Восстановлено 2026-11-05

- `build/`: 9 файлов, 1,52 МБ. Активные `combat.dll`, `combat_ai.dll`,
  `fractal.dll`, подготовленные `combat_rules.h` / `cp_geometry.inc`, метаданные
  `combat_rules.json` / `native_sources.json`. Две дополнительные исторические
  DLL нужны для неизменённого списка хешей уже выполненного CP reset study;
  активный планировщик их не использует.
- `artifacts/rl/`: 1106 файлов, 48,89 МБ. Manifest корпуса, 320 открытых карт
  10×10 и 228 training-карт 16×16; первые 512 демонстраций исходного CP-корпуса
  для фиксированного GPU-бенчмарка; initial и baseline more-data best;
  небольшие отчёты, метрики и сведения о происхождении родительских опытов.
- Уже находившиеся на диске `cp-imitation-own-states/`, `cp-reset-study/` и
  `gru-ablation-20261105/` сохранены без перезаписи.

Точный перечень и CRC записаны в
`artifacts/rl/archive-recovery/selection.json`, проверки — в `verification.json`.
SHA256 512 входных эпизодов, baseline-checkpoint и файлов CP study сверены с
сохранёнными manifests. Карты закрытого test не извлекались.

Из корня проекта повторное извлечение выбранного минимума возможно по локальным
спискам, без перезаписи уже существующих файлов:

```powershell
& 'C:\Program Files\7-Zip\7z.exe' x build/build-history.7z -obuild '-i@artifacts/rl/archive-recovery/extract-build.txt' -scsUTF-8 -aos
& 'C:\Program Files\7-Zip\7z.exe' x artifacts/rl/development-history-2.7z -oartifacts/rl '-i@artifacts/rl/archive-recovery/extract-rl.txt' -scsUTF-8 -aos
```

## Текущие локальные результаты

- Итоговая own-states policy:
  `artifacts/rl/cp-imitation-own-states/cycle-02/checkpoints/step-0500.pt`.
- Родитель и вход скорости:
  `artifacts/rl/cp-imitation-more-data/checkpoints/best.pt`.
- [CP со сбросом против обычного CP](../artifacts/rl/cp-reset-study/report.md).
- [Скорость с GRU и без неё](../artifacts/rl/gru-ablation-20261105/report.md).
- [Общий локальный индекс](../artifacts/rl/README.md).

Полные исторические обучающие данные, промежуточные модели, старые сборки,
реплеи родительских опытов и тяжёлые nsys/ncu-трассы остаются в архивах.
Минимум достаточен для текущих матчей, CP reset study и фиксированного
GPU-бенчмарка. Повтор полного исторического collect/train требует отдельно
восстановить все его входы; строгие проверки происхождения не ослаблялись.
