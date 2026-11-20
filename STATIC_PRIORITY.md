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

## Динамический приоритет на инференсе

Для `run_layout_suite` режим включается флагом `--static_priority dynamic`. Для
`eval_static_priority` добавьте `dynamic` в `--modes`. Оба запуска используют
одинаковые параметры динамического приоритета.

| `--dynamic_priority_metric` | Кому уступают |
| --- | --- |
| `distance` (по умолчанию) | Дрону с большим расстоянием до исходной цели |
| `stall` | Дрону, который дольше не сократил расстояние до цели хотя бы на 5 см. В радиусе 0,3 м счётчик сбрасывается |
| `critic` | Дрону с меньшим \(V(s)\). Критик оценивает наблюдение с исходной целью на том же шаге |

При равном приоритете старше дрон с меньшим номером. Для соседей из штатных
слотов прогнозируется минимальное расстояние за ближайшие 0,5 с при постоянной
относительной скорости. Если оно меньше 1,75 порога столкновения симулятора,
младший получает цель зависания в точке, где началось уступание, на 0,2 с.
Параметры задаются через `--dynamic_priority_lookahead`, `--dynamic_priority_conflict_scale`
и `--dynamic_priority_hold`. Меняются первые три координаты входа актора.
Награда, метрики и критик при расчёте приоритета используют исходную цель.

Если младший сосед приближается к точке зависания уступающего дрона, он может
унаследовать его приоритет. Передача идёт по направлению убывания базового
приоритета. Это исключает циклы и позволяет провести наследование по цепочке
одним обходом графа. На графе готовых наблюдений контроллер требует \(O(NK)\)
времени и памяти. Поиск соседей и расчёт столкновений в симуляторе используют
попарные расстояния. Зависание через замороженную политику проверяется
экспериментально и может менять траекторию и время долёта.

### Парный прогон

```sh
export PYGLET_HEADLESS=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
config=experiments/swarm100-1b/config.json
checkpoint=experiments/swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth
python scripts/download_pretrained.py --output "$(dirname "$checkpoint")"

for mode in off mask dynamic; do
  python -m swarm_rl.run_layout_suite \
    --checkpoint "$checkpoint" --config "$config" \
    --layouts random_baseline open_room --agents 8 100 --episodes8 16 --episodes 24 \
    --static_priority "$mode" --no_diagnostics --output "results/dynamic-study/$mode"
done

python scripts/compare_static_priority.py \
  --runs off=results/dynamic-study/off mask=results/dynamic-study/mask dynamic=results/dynamic-study/dynamic \
  --output results/dynamic-study/comparison.csv
```

Для `stall` и `critic` задайте другую папку вывода для каждой метрики. В
`episodes.csv` у динамических режимов есть доля шагов уступания и доля шагов
с фактическим повышением приоритета. Сиды совпадают между режимами, в том числе
для генераторов стартового разброса отдельных дронов. Сравнение сверяет хеши
геометрии, вычисляет разности по каждому сиду и строит 95%-е процентильные
бутстреп-интервалы по эпизодам. Для времени долёта берутся только пары, где
в обоих эпизодах хотя бы один дрон достиг цели. Число таких пар указано в
колонках с суффиксом `_n`.
