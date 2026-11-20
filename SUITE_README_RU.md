# QuadSwarm: 18 схем и оценка модели на 1 млрд шагов

## Состав архива

- `source/`: исходный код с изменениями, один checkpoint и исходный config.
- `changed_files/`: только добавленные и изменённые файлы с сохранёнными путями.
- `patches/from_swarm100.patch`: изменения относительно swarm100.
- `analysis/`: PDF, Excel, таблицы CSV, графики PNG/SVG, HTML без интернета.
- `analysis/layouts/`: атлас всех схем, изображения и координаты JSON.
- `runs_15s/`: основные завершённые эпизоды, сырые CSV и TensorBoard.
- `runs_30s/`: отдельная диагностическая проверка длинного горизонта.
- `validation/`: логи тестов и проверка полноты результатов.

Ветка подготовлена локально. Автоматической публикации на GitHub нет.

## Источники

Код и конфигурация: https://github.com/Vikvins/quad-swarm-rl/tree/swarm100
commit `85a0a2e1ff9993929b95d52646385a3b36fc2289`.

Диагностики: https://github.com/Vikvins/quad-swarm-rl/tree/experiment-attention-katya
commit `280800367d868d888424a1cab573f98a2d1e87d6`.
Их исходные версии находятся в `references/`.

Модель: `checkpoint_000781254_1000005120.pth`, 1 000 005 120 шагов.
SHA256: `f429fa60b70a82ba1aaf54ebbdd0f966d65995d46caf87e0fd8628e1f15b2524`.
Веса не обновляются.

## Геометрия

Исходная случайная среда, открытая комната и восемь пар:
прямой, узкий, Г-образный, S-образный, сужающийся проход,
широкая U-камера, Т-развилка, проход с боковыми ответвлениями.
В каждой паре старты, цели и seed совпадают; добавляются только закрывающие стены.
Все новые карты проверены на существование свободного пути.

Туннели представлены стенами на всю высоту комнаты, то есть 2.5D.
Крыша и новый 3D-сенсор не добавлены. Вход модели сохраняет размерность 40.

## Перенос в свой форк

Создайте отдельную ветку от актуальной swarm100. При активном обучении используйте
отдельный Git worktree. Для ручного переноса берите файлы из `changed_files/`.
Перед применением патча проверьте `git apply --check patches/from_swarm100.patch`.

Окружение: Python 3.11, PyTorch 2.5.x, Sample Factory 2.1.1,
NumPy 1.26.4, Numba 0.60.0. Можно использовать существующее рабочее окружение Mac.
Все следующие команды выполняются из корня проекта.

## Проверки

```bash
export PYGLET_HEADLESS=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QUAD_PRETRAINED_CHECKPOINT="$PWD/experiments/swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth"
python -m pytest tests/ -q
```

Для тестов требуется pytest. Исходные `python -m unittest` содержат 9 ошибок
старых интерфейсов и проверены отдельно. Они не выдаются за прошедшие.

## Основная серия

```bash
python -m swarm_rl.run_layout_suite \
  --checkpoint experiments/swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth \
  --config experiments/swarm100-1b/config.json \
  --output results/suite_15s \
  --agents 8 100 --episodes8 16 --episodes 8 --duration 15 --workers 4
```

Это 432 эпизода. На Mac начните с `--workers 2`.
RNG NumPy и Numba явно инициализируются для каждого seed.
Завершённые эпизоды определяются по `episode.json` и не пересчитываются при повторе.
После прерывания переместите незавершённые папки без `episode.json` перед повтором.

## Метрики через стандартный enjoy

```bash
python scripts/prepare_eval_experiment.py
python drone_influence.py --env=quadrotor_multi --train_dir=./train_dir \
  --experiment=swarm100-1b-eval --device=cpu --eval_deterministic=True \
  --no_render --quads_render=False --env_gpu_observations=False \
  --metrics-dir=./metrics_out/random_run --max_num_episodes=2
```

Для новой схемы добавьте:
`--obstacle_layout=suite --layout_id=straight_deadend --layout_seed=0`
и выберите новую папку `--metrics-dir`.

Основной сборщик вызывает те же классы диагностик напрямую, добавляет навигационные
метрики и фиксирует seed Numba. Для воспроизводимого сравнения используйте его.

Формулы исходных CSV сохранены. В drone_influence исправлена поддержка имени
`all_neighbor_obs_dim` наряду с `all_neighbor_obs_size`.
Дополнительно настраивается объём TensorBoard через METRICS_TB_EVERY и METRICS_TB_AGENTS.
В основной серии CSV содержат все дроны на каждом шаге, а TensorBoard - средние
по рою каждые 10 шагов. Чувствительность измеряется 8 пробами каждые 10 шагов.

```bash
tensorboard --logdir results/suite_15s
```

## Атлас и анализ

```bash
python scripts/build_suite_atlas.py --output results/layouts
python scripts/analyze_layout_suite.py --input results/suite_15s --output results/analysis
```

Для включения случайной карты в новый атлас передайте `--baseline-json` из эпизода
`runs_15s/n100/random_baseline/seed000/layout.json`. Готовый PDF содержит все 18 схем.
Для отчёта нужны pandas, matplotlib, reportlab. DejaVu Sans автоматически находится
через Matplotlib, включая macOS. Эксперименты от PDF и reportlab не зависят.

`summary.csv` содержит все схемы. `paired_differences.csv` содержит парные разницы.
В `baseline_comparison.csv` исходная случайная среда сравнивается с новыми картами,
где старты и цели другие. Это контекстное, а не строго парное сравнение.

Bootstrap выполняется по целым эпизодам. Дроны внутри эпизода не считаются
независимыми повторами. Нулевая чувствительность не означает уверенность.
Основная метрика успеха едина во всех картах: радиус 0.3 м, удержание 0.5 с,
без контактов за весь эпизод. Сохранённый дополнительно успех исходного Env
использует собственные пороги сценариев: 0.5 м в случайном и 0.3 м в новых.
Его нельзя подменять основной единой метрикой при сравнении разных сценариев.
`obstacle_share` не является attention или причинной атрибуцией.
Время вычислений измерялось под параллельной нагрузкой и не является аппаратным бенчмарком.
