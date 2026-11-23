# Канал приоритета только при обучении — 100 дронов

Политика училась выдавать число намерения соседям и выбирать, уступать ли им. На обучении канал периодически отключали на весь эпизод. В валидации канал был выключен во всех эпизодах этого режима. Лётная политика заморожена.

Обучение завершилось на **1 000 000 200 шагах агентов**. Использованы шесть параллельных сцен на 100 дронах, по три экземпляра узкого коридора и парного тупика. Запуск выполнен на облачной RTX 3070.

[Лётный чекпоинт](../swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth) · [Конфиг](../swarm100-1b/config.json) · [Код обучения](../priority-coordinator-1b/train_priority.py) · [Сводка обучения](training-summary.json)

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

## Валидация

Для каждого размера роя использовали 8 проходных коридоров и 8 парных тупиков. Во всех трёх режимах совпадали сцены, стартовые позиции (seed 50000), лётные веса и лимит 15 секунд. Датчики работали штатно. Долёт засчитывали, если дрон удерживался в радиусе 0,3 м от цели не менее 0,5 с и за весь эпизод не было контактов.

| Сцены | Базовая модель | Канал выключен | Статический фильтр |
|---|---:|---:|---:|
| Коридоры, 8 дронов | 37/64 · 57,8% | 37/64 · 57,8% | **46/64 · 71,9%** |
| Коридоры, 100 дронов | 123/800 · 15,4% | 142/800 · 17,8% | **186/800 · 23,3%** |
| Тупики, 8 дронов | 0/64 | 0/64 | 0/64 |
| Тупики, 100 дронов | 0/800 | 0/800 | 0/800 |

На 100 дронах политика с выключенным каналом дала на 19 безаварийных долётов больше базовой модели. Статический фильтр дал на 44 долёта больше политики с выключенным каналом.

[Результаты 96 эпизодов](results/episodes.csv) · [Сводка метрик](results/summary.csv) · [Параметры и контрольные суммы](results/manifest.json) · [Конфигурации сцен](https://github.com/Vikvins/quad-swarm-rl/blob/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20/gym_art/quadrotor_multi/obstacles/suite_layouts.py)

## Запуск валидации

Из корня ветки `swarm100` в окружении QuadSwarm. Переменная `QUADSWARM_SIM` указывает на checkout [симулятора](https://github.com/Vikvins/quad-swarm-rl/tree/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20).

```bash
base=experiments/swarm100-1b
PYTHONPATH="$QUADSWARM_SIM" python experiments/priority-coordinator-1b/evaluate_priority.py \
  --config "$base/config.json" \
  --flight "$base/checkpoints/checkpoint_000781254_1000005120.pth" \
  --peer experiments/priority-channel-1b/priority.pt \
  --output /tmp/quads-peer-validation --workers 20 --modes peer_off baseline static_brake
```
