# Внешний координатор приоритета — 100 дронов

Модуль приоритета обучен до **1 000 000 200 шагов агентов** на RTX 4070 Ti SUPER. Лётная политика заморожена. Обучение проходило в шести параллельных сценах на 100 дронах, по три экземпляра узкого коридора и парного тупика. [Лётный чекпоинт](../swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth) · [Конфиг лётной политики](../swarm100-1b/config.json) · [Финальные веса координатора](priority.pt) · [Ход обучения](training-summary.json)

## Веса

| Шаги обучения | Чекпоинт |
|---:|---|
| 100 032 000 | [checkpoint_000100032000.pt](checkpoints/checkpoint_000100032000.pt) |
| 200 025 600 | [checkpoint_000200025600.pt](checkpoints/checkpoint_000200025600.pt) |
| 300 019 200 | [checkpoint_000300019200.pt](checkpoints/checkpoint_000300019200.pt) |
| 400 012 800 | [checkpoint_000400012800.pt](checkpoints/checkpoint_000400012800.pt) |
| 500 006 400 | [checkpoint_000500006400.pt](checkpoints/checkpoint_000500006400.pt) |
| 600 000 000 | [checkpoint_000600000000.pt](checkpoints/checkpoint_000600000000.pt) |
| 700 032 000 | [checkpoint_000700032000.pt](checkpoints/checkpoint_000700032000.pt) |
| 800 025 600 | [checkpoint_000800025600.pt](checkpoints/checkpoint_000800025600.pt) |
| 900 019 200 | [checkpoint_000900019200.pt](checkpoints/checkpoint_000900019200.pt) |
| 1 000 000 200 | [checkpoint_001000000200.pt](checkpoints/checkpoint_001000000200.pt) |

## Тест

Для каждого размера роя использовали восемь коридоров и восемь парных тупиков, по одному набору стартов на геометрию (сид 50000). Три режима прошли одинаковые сцены за лимит 15 с, с тем же лётным чекпоинтом. Датчики работали штатно. Всего 96 эпизодов. Статический фильтр проверяет все пары и тормозит по приоритету номера. Координатор применяет выученную оценку к конфликтам с двумя соседями.

В таблице — доля дронов, долетевших без контактов с препятствиями, другими дронами или границами комнаты.

| Дронов | Сцены | База | Статическое торможение | Координатор |
|---:|---|---:|---:|---:|
| 8 | Коридоры | 57,8% | 71,9% | 62,5% |
| 8 | Тупики | 0,0% | 0,0% | 0,0% |
| 100 | Коридоры | 15,75% | 22,75% | 22,25% |
| 100 | Тупики | 0,0% | 0,0% | 0,0% |

[Сводка](results/summary.csv) · [Эпизоды](results/episodes.csv) · [Настройки и хеши](results/manifest.json) · [Обучение](train_priority.py) · [Тест](evaluate_priority.py) · [Симулятор, 7f4596e](https://github.com/Vikvins/quad-swarm-rl/tree/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20)

### Запуск теста

`QUADSWARM_SIM` указывает на checkout симулятора по ссылке выше. Запусти из корня ветки `swarm100` в окружении QuadSwarm.

```bash
run=experiments/priority-coordinator-1b
base=experiments/swarm100-1b
PYTHONPATH="$QUADSWARM_SIM" PYGLET_HEADLESS=true MPLBACKEND=Agg \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python "$run/evaluate_priority.py" --config "$base/config.json" \
  --flight "$base/checkpoints/checkpoint_000781254_1000005120.pth" \
  --coordinator "$run/priority.pt" --workers 16 --output /tmp/quad-priority-eval
```
