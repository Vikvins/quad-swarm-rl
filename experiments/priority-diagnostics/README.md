# Диагностика приоритета и старта

## Канал и первые контакты

Повтор 64 эпизодов на 100 дронах с записью времени первого контакта. На одном наблюдении подмена чисел от соседей меняла решение об уступании в 0,008% случаев в коридорах и в 0,042% в тупиках при включённом канале.

В узком коридоре при общем старте первый контакт был через 0,45 с. За первую секунду с соседом столкнулись 42 дрона, по рядам с интервалом 250 мс — 15. За 15 с без столкновений долетели 10 и 12 дронов соответственно.

[Замеры канала](results/diagnostic-summary.txt) · [Очередь, скорости и видимость перед контактом](results/stagger-summary.txt) · [Схема стартов](results/narrow-starts-and-contacts.png) · [Видео координатора](results/narrow-open-100-coordinator.mp4)

[Ранг пары перед контактом](results/nearest-pair-coverage.csv) · [Первые контакты и флаги уступания](results/coordinator-first-pair-contacts.csv)

## Встречный поток и Т-развилка

Во встречном потоке две сцены по 100 дронов с двумя направлениями. По рядам выпускали с интервалом 250 мс. В высотном варианте одну группу поднимали на 3,3 м. Вариант с подъёмом в полёте начинал движение по рядам после выхода большей части верхней группы на заданную высоту или после 2 с. В таблице суммы по двум сценам.

| Режим | Контакт с соседом до 1 с | До 1,5 с | Безаварийный долёт |
|---|---:|---:|---:|
| Все сразу | 20 | 45 | 93 |
| По рядам | 14 | 24 | **102** |
| Высоты заданы на старте | 20 | 44 | 65 |
| Высоты и очередь | 8 | 15 | 93 |
| Подъём, затем очередь | **6** | **7** | 79 |

На Т-развилке в одной сцене на 100 дронов цели назначались в левый и правый рукава.

| Режим | Контакт с соседом до 1 с | До 1,5 с | Безаварийный долёт |
|---|---:|---:|---:|
| Все сразу | 21 | 48 | 8 |
| По рядам | 6 | 35 | **11** |
| Высоты и очередь | **2** | **29** | 9 |

[Результаты по всем сценам](results/heights/summary.csv) · [Замеры по дронам и эпизодам](results)

### Запуск

`QUADSWARM_SIM` указывает на checkout [симулятора `7f4596e`](https://github.com/Vikvins/quad-swarm-rl/tree/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20). Используется [полётная модель на 1 млрд](../swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth).

```bash
export PYTHONPATH="$PWD/experiments/priority-coordinator-1b:$QUADSWARM_SIM"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
run=experiments/priority-diagnostics
python "$run/diagnose.py" --output /tmp/quads-channel-trace
python "$run/diagnose_starts.py" --output /tmp/quads-start-trace
python "$run/diagnose_starts.py" --disable-downwash --output /tmp/quads-start-no-downwash
python "$run/height_start.py" --output /tmp/quads-height-results
python "$run/fork_start.py" --output /tmp/quads-fork-results
```
