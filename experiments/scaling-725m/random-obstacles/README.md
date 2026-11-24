# 200 и 300 дронов со случайными препятствиями

[Чекпоинт 725 738 240](../../swarm100-1b/checkpoints/checkpoint_000566983_725738240.pth), обученный на 100 дронах. На ПК проверили 16 сцен на 200 и 300 дронах. В каждой сцене сравнили три режима на одних и тех же стартах, целях и препятствиях. При отказе у четверти дронов замораживались девять показаний обзора препятствий на 0,5 или 2 с. Сиды 50000–50015.

Свободные клетки прежней карты получили три высотные позиции для стартов и целей. Размер комнаты, плотность препятствий и диапазон высот 1–3 м сохранены. При достаточном числе клеток используется исходный алгоритм.

В ячейках доля всех дронов, долетевших без столкновений, и доля долётов среди дронов, выбранных для отказа. В штатном режиме для второй доли используется та же группа дронов.

| Дронов | Штатный датчик | Заморозка 0,5 с | Заморозка 2 с |
|---:|---:|---:|---:|
| 200 | 84,9% / 85,8% | 84,2% / 84,0% | 83,4% / 81,6% |
| 300 | 78,7% / 80,3% | 77,6% / 78,3% | 79,1% / 75,9% |

Весь рой долетел без столкновений в 0 из 16 сцен каждого режима.

[Сводка](results/summary.csv) · [48 эпизодов на 200](results/n200-episodes.csv) · [48 эпизодов на 300](results/n300-episodes.csv)

### Запуск

Из корня ветки `swarm100`. `QUADSWARM_SIM` указывает на [симулятор `7f4596e`](https://github.com/Vikvins/quad-swarm-rl/tree/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20).

```bash
base=experiments/swarm100-1b
printf '%s\n' "$base/checkpoints/checkpoint_000566983_725738240.pth" > /tmp/quads-random-checkpoint.txt
for n in 200 300; do
  PYTHONPATH="$PWD/experiments/swarm8-1b:$QUADSWARM_SIM" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python experiments/scaling-725m/random-obstacles/evaluate_capacity.py --config "$base/config.json" \
      --manifest /tmp/quads-random-checkpoint.txt --agents "$n" --episodes 16 --workers 12 \
      --output "/tmp/quads-random-n$n.csv"
done
```
