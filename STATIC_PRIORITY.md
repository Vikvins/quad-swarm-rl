# Статическая приоритизация без переобучения

Приоритет задаётся номером дрона: дрон 0 самый старший, дрон N-1 самый младший. Сеть не меняется и не дообучается, меняется только то, каких соседей и как она видит.

## Переключатель

`--quads_static_priority`:

- `off` — без приоритета, исходное поведение (значение по умолчанию);
- `mask` — дрон видит только старших соседей (два ближайших среди дронов с меньшим номером), младших для него нет. Если старших меньше двух (у дрона 0 их нет), в пустой слот кладётся «никого рядом»: сосед на максимальной дистанции с нулевой относительной скоростью;
- `soft` — дрон видит двух ближайших соседей как обычно, но старшего соседа ему показывают ближе, чем он есть: относительная позиция умножается на `--quads_static_priority_scale` (по умолчанию 0.7).

Флаг работает в любой команде, которая создаёт среду, например при просмотре полёта:

```sh
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --train_dir=... --experiment=... \
    --quads_mode=o_swap_goals --quads_static_priority=mask
```

## Сравнение режимов

`swarm_rl/eval_static_priority.py` прогоняет модель в нескольких режимах на одинаковых стартах, целях и препятствиях и считает долёты без столкновений, столкновения и время до цели, в том числе по группам номеров дронов.

С приоритетом и без:

```sh
python -m swarm_rl.eval_static_priority \
    --cfg config.json --checkpoint checkpoint_000781254_1000005120.pth \
    --num_agents 8 100 --episodes 10 --scenario o_swap_goals \
    --modes off mask --out results_swap
```

Только с приоритетом: `--modes mask`, только без: `--modes off`, все три: `--modes off mask soft`.

`--scenario` — название сценария (`quads_mode`), например `o_random`, `o_swap_goals` или новые коридор и тупик. Результаты сохраняются в папку `--out`: `summary.csv` (средние по режимам), `by_rank.csv` (по группам номеров, группа 1 — самые старшие), `episodes.csv` и `drones.csv`.

## Коридоры и тупики

`swarm_rl/run_layout_suite.py` принимает тот же переключатель: `--static_priority off|mask|soft` и `--static_priority_scale`. Каждый режим запускается в свою папку `--output`. Готовые эпизоды при повторном запуске не пересчитываются, поэтому запуск другого режима в папку с результатами прошлого режима останавливается с ошибкой.

```sh
python -m swarm_rl.run_layout_suite \
  --checkpoint experiments/swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth \
  --config experiments/swarm100-1b/config.json \
  --output results/suite_off --static_priority off \
  --agents 8 100 --episodes8 16 --episodes 8 --duration 15 --workers 2

python -m swarm_rl.run_layout_suite \
  --checkpoint experiments/swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth \
  --config experiments/swarm100-1b/config.json \
  --output results/suite_mask --static_priority mask \
  --agents 8 100 --episodes8 16 --episodes 8 --duration 15 --workers 2
```

Сравнение по схемам на одних и тех же сидах (старты, цели и стены совпадают):

```sh
python scripts/compare_static_priority.py --runs off=results/suite_off mask=results/suite_mask \
    --output results/priority_comparison.csv
```

Режим записывается в `manifest.json`, в `effective_config.json` каждого эпизода и в колонку `static_priority` файла `episodes.csv`. Анализ одной серии (`scripts/analyze_layout_suite.py`) запускается для каждой папки отдельно, как раньше.

## Тесты

```sh
export PYGLET_HEADLESS=true OMP_NUM_THREADS=1
python -m pytest tests/ -q
```

`tests/test_static_priority.py` проверяет, что `off` совпадает с исходным наблюдением, `mask` показывает только старших соседей, `soft` меняет только позиции старших, флаг доходит до среды в случайной среде, коридоре и тупике, а сравнение берёт только общие эпизоды.
