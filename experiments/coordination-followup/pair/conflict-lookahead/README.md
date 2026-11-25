# Внешний координатор с выбором уступания

При прогнозируемом конфликте на горизонте 0,65 с фильтр сравнивает продолжение полёта, торможение первого дрона и торможение второго. Для каждой траектории он оценивает прогнозируемые контакты с другими дронами, препятствиями и границами комнаты. При равном улучшении уступает дрон с меньшим рангом. Использованы [лётные веса на 1 млрд](../../../swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth) и [прежние веса координатора на 100 млн](../../../pair-priority-100m/frozen/checkpoint_000100000000.pt).

## Результаты

Восемь коридоров на каждом сиде, 100 дронов в сцене, штатные датчики и 15 с симуляции. В каждой паре «коридор, сид» старты, цели и препятствия совпадали во всех режимах. Безаварийный долёт засчитывался после 0,5 с в радиусе 0,3 м от цели при отсутствии контактов за весь эпизод.

| Сид | Прежний внешний с прогнозом к цели | Канал с фильтром | Внешний с выбором уступания |
|---|---:|---:|---:|
| 50000 | 240/800 | 258/800 | **315/800** |
| 50001 | 221/800 | 257/800 | **300/800** |
| 50002 | 223/800 | 218/800 | **287/800** |
| **Итого** | **684/2400 (28,5%)** | **733/2400 (30,5%)** | **902/2400 (37,6%)** |

Для сида 50000 на ПК заново прогнали оба внешних варианта. Результат канала взят из [сопоставимых эпизодов](../../channel/messages/episodes.csv), хеши сцен совпадают. Для сидов 50001 и 50002 все три режима прогнали на ПК. В зигзагообразном коридоре у всех режимов было 0 безаварийных долётов на трёх сидах.

[Сводка 50000](results/comparison.csv) · [16 эпизодов](results/episodes.csv) · [Манифест](results/manifest.json)

[Сводка 50001–50002](new-starts/comparison.csv) · [48 эпизодов](new-starts/episodes.csv) · [Манифест](new-starts/manifest.json)

## Запуск

Из корня `swarm100` в окружении с зависимостями проекта. `QUADSWARM_SIM` указывает на checkout [симулятора `7f4596e`](https://github.com/Vikvins/quad-swarm-rl/tree/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20).

```bash
base=experiments/swarm100-1b
report=experiments/coordination-followup
run=$report/pair/conflict-lookahead
flight=$base/checkpoints/checkpoint_000781254_1000005120.pth
pair=experiments/pair-priority-100m/frozen/checkpoint_000100000000.pt
peer=experiments/priority-channel-100m/checkpoint_000100000000.pt
export PYTHONPATH="$PWD/experiments/priority-coordinator-1b:$PWD/experiments/priority-diagnostics:$QUADSWARM_SIM"
export PYGLET_HEADLESS=true MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python "$run/evaluate_conflict_lookahead.py" --config "$base/config.json" --flight "$flight" \
  --coordinator "$pair" --reference "$report/pair/horizon" --output /tmp/quads-lookahead-50000
python "$run/validate_new_starts.py" --config "$base/config.json" --flight "$flight" \
  --coordinator "$pair" --peer "$peer" --output /tmp/quads-lookahead-50001-50002
```

Манифесты сохраняют хеши файлов исходных прогонов. Опубликованные скрипты принимают пути к файлам из репозитория.
