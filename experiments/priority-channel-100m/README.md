# Усиленный канал приоритета

**100 млн шагов агентов**, 100 дронов, 32 параллельные среды. Сцены обучения — узкий коридор и парный тупик. Полётная политика взята из [чекпоинта на 1 млрд](../swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth) и заморожена.

Числа приоритета перед решением об уступании умножали на 2. Дополнительная функция потерь с коэффициентом 0,2 учила согласовывать приоритеты в прогнозируемых конфликтах и штрафовала случай, когда полученное сообщение не повышало уверенность в нужном решении относительно занулённых сообщений.

[Веса на 100 млн](checkpoint_000100000000.pt) · [Сводка обучения](training-summary.json) · [Функция потерь](../priority-coordinator-1b/peer_channel_objective.py) · [Валидация на 8 и 100 дронах, проверка сообщений и веса до 200 млн](../coordination-followup/README.md).

### Запуск

`QUADSWARM_SIM` указывает на checkout [симулятора `7f4596e`](https://github.com/Vikvins/quad-swarm-rl/tree/7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20). Запуск из корня ветки `swarm100` с CUDA-окружением.

```bash
base=experiments/swarm100-1b
PYTHONPATH="$QUADSWARM_SIM" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python experiments/priority-coordinator-1b/train_priority.py peer --config "$base/config.json" \
    --flight-checkpoint "$base/checkpoints/checkpoint_000781254_1000005120.pth" \
    --agents 100 --arenas 32 --train-steps 100000000 --checkpoint-steps 1000000 \
    --peer-message-gain 2 --peer-use-loss 0.2 --output /tmp/quads-peer-weighted
```
