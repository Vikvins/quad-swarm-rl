"""PPO for sparse pair observations and optional joint flight training."""

import json
import time

import gymnasium as gym
import numpy as np
import torch

from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.evaluate_layouts import load_checkpoint

from pair_priority import PairController, PairPriority, drone_features


def advantage_targets(rewards, values, alive, next_value, gamma, gae_lambda):
    advantages = torch.empty_like(rewards)
    gae = torch.zeros_like(next_value)
    for tick in reversed(range(len(values))):
        delta = rewards[tick] + gamma * next_value * alive[tick] - values[tick]
        gae = delta + gamma * gae_lambda * alive[tick] * gae
        advantages[tick] = gae
        next_value = values[tick]
    returns = (values + advantages).flatten()
    advantages = advantages.flatten()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    return returns, advantages


def optimize(policy, optimizer, rollout, next_value):
    features, pairs, actions, old_log_prob, decisions, values, rewards, alive = (
        torch.stack(items) for items in zip(*rollout)
    )
    returns, advantages = advantage_targets(rewards, values, alive, next_value, .99, .95)
    features, pairs, actions, old_log_prob, decisions = (
        item.flatten(0, 1) for item in (features, pairs, actions, old_log_prob, decisions)
    )
    for _ in range(2):
        for rows in torch.randperm(len(returns), device=features.device).split(96):
            distribution, value = policy(features[rows], pairs[rows])
            ratio = (distribution.log_prob(actions[rows]) - old_log_prob[rows]).clamp(-20, 20).exp()
            advantage = advantages[rows, None]
            clipped = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
            mask = decisions[rows]
            actor_loss = -(clipped * mask).sum() / mask.sum().clamp_min(1)
            loss = actor_loss + .5 * (value - returns[rows]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
    return float(loss.detach())


def optimize_flight(flight, optimizer, rollout, next_value, cfg, rnn_size):
    observations, actions, old_log_prob, values, rewards, alive = (
        torch.stack(items) for items in zip(*rollout)
    )
    returns, advantages = advantage_targets(rewards, values, alive, next_value, cfg.gamma, cfg.gae_lambda)
    observations, actions = observations.flatten(0, 2), actions.flatten(0, 2)
    old_log_prob = old_log_prob.flatten()
    for _ in range(cfg.num_epochs):
        for rows in torch.randperm(len(returns), device=observations.device).split(cfg.batch_size):
            states = observations.new_zeros((len(rows), rnn_size))
            head = flight.forward_head({"obs": observations[rows]})
            core, _ = flight.forward_core(head, states)
            outputs = flight.forward_tail(core, values_only=False, sample_actions=False)
            current = flight.action_distribution().log_prob(actions[rows])
            ratio = (current - old_log_prob[rows]).clamp(-20, 20).exp()
            clip = cfg.ppo_clip_ratio
            advantage = advantages[rows]
            clipped = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
            value_loss = (outputs["values"] - returns[rows]).square().mean()
            loss = -clipped.mean() + cfg.value_loss_coeff * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flight.parameters(), cfg.max_grad_norm)
            optimizer.step()
    return float(loss.detach())


def train_pair(args, connections, first, cfg, resume, save_checkpoint, stack_states):
    progress = resume["progress"] if resume else {}
    update = progress.get("updates", 0)
    initial_update = update
    torch.manual_seed(args.seed + update)
    torch.set_num_threads(1)
    device = torch.device("cuda")

    obs_space, action_space, dt, threshold, _ = first[0]
    state = stack_states([item[-1] for item in first])
    swarm_shape = state[0].shape[:2]
    arena_count, agent_count = swarm_shape
    steps_per_tick = arena_count * agent_count
    flight = create_actor_critic(cfg, gym.spaces.Dict({"obs": obs_space}), action_space)
    if not args.train_flight:
        flight.load_state_dict(load_checkpoint(args.flight_checkpoint)["model"], strict=True)
    flight.eval().to(device).requires_grad_(args.train_flight)
    flight_optimizer = torch.optim.Adam(flight.parameters(), lr=cfg.learning_rate) if args.train_flight else None
    rnn_size = get_rnn_size(cfg)
    recurrent = torch.zeros(steps_per_tick, rnn_size, device=device)
    policy = PairPriority().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    if resume:
        policy.load_state_dict(resume["policy"])
        optimizer.load_state_dict(resume["optimizer"])
        if flight_optimizer is not None:
            flight.load_state_dict(resume["model"], strict=True)
            flight_optimizer.load_state_dict(resume["flight_optimizer"])

    controller = PairController(arena_count, agent_count, dt, threshold, args)
    episodes = progress.get("episodes", 0)
    total_steps = progress.get("steps", 0)
    total_decisions = progress.get("decisions", 0)
    next_save = (total_steps // args.checkpoint_steps + 1) * args.checkpoint_steps
    next_milestone = (total_steps // args.milestone_steps + 1) * args.milestone_steps
    latest = args.output / "latest.pt"
    rates = []
    measure_begin = time.perf_counter()
    measure_steps = total_steps
    inside = np.zeros(swarm_shape, dtype=np.int64)
    arrived = np.zeros(swarm_shape, dtype=bool)
    arrival_ticks = max(1, round(.5 / dt))
    verified_joint_update = False

    def snapshot(state, ended):
        _, pos, vel, goals, _ = state
        pairs, active, sample = controller.observe(pos, vel, ended)
        features = torch.as_tensor(drone_features(pos, vel, goals), device=device)
        return features, torch.as_tensor(pairs, device=device), active, sample

    def training_progress():
        return dict(steps=total_steps, updates=update, episodes=episodes, decisions=total_decisions)

    def checkpoint(path):
        state = {}
        if flight_optimizer is not None:
            state = dict(model=flight.state_dict(), flight_optimizer=flight_optimizer.state_dict())
        save_checkpoint(path, policy, optimizer, args, training_progress(), **state)

    def flight_input(observations, offsets):
        flight_obs = torch.as_tensor(observations, device=device).clone()
        flight_obs[:, :, :3] += torch.as_tensor(offsets, device=device)
        return flight.normalize_obs({"obs": flight_obs.flatten(0, 1)})

    observation = snapshot(state, np.ones(arena_count, dtype=bool))
    while args.train_steps is None or total_steps < args.train_steps:
        warmed_up = update - initial_update >= args.warmup_updates
        if update - initial_update == args.warmup_updates:
            measure_begin, measure_steps = time.perf_counter(), total_steps
        begun = time.perf_counter()
        ticks = args.rollout
        if args.train_steps is not None:
            ticks = min(ticks, (args.train_steps - total_steps + steps_per_tick - 1) // steps_per_tick)
        rollout = []
        flight_rollout = []
        sampled = 0
        waits = 0
        arrivals = 0
        for _ in range(ticks):
            observations, pos, vel, goals, _ = state
            features, pairs, active, sample = observation
            with torch.no_grad():
                distribution, value = policy(features, pairs)
                ranks = distribution.sample()
                sampled += int(sample.sum())
                offsets = controller.offsets(ranks.cpu().numpy(), sample, active, pos, vel, goals)
                waits += int(controller.previous_active.sum())
                normalized = flight_input(observations, offsets)
                flight_outputs = flight(normalized, recurrent)
                actions = flight_outputs["actions"] if args.train_flight else flight.action_distribution().means
                commands = actions.cpu().numpy().reshape(*swarm_shape, -1)
                commands = np.clip(commands, action_space.low, action_space.high)

            for connection, command in zip(connections, commands):
                connection.send(command)
            results = [connection.recv() for connection in connections]
            ended = np.array([result[2] for result in results])
            end_positions = np.stack([result[3] for result in results])
            end_distance = np.linalg.norm(end_positions - goals, axis=-1)
            inside = np.where(end_distance <= .3, inside + 1, 0)
            arrived_now = (inside >= arrival_ticks) & ~arrived
            arrived |= arrived_now
            arrivals += int(arrived_now.sum())
            progress_reward = np.linalg.norm(pos - goals, axis=-1) - end_distance
            wait_penalty = .25 * dt * controller.previous_active
            sim_reward = np.stack([result[1] for result in results])
            shaped_reward = sim_reward + progress_reward + 2 * arrived_now - wait_penalty
            reward = shaped_reward.mean(axis=1, dtype=np.float32)
            inside[ended] = 0
            arrived[ended] = False
            state = stack_states([result[0] for result in results])
            observation = snapshot(state, ended)
            alive = torch.as_tensor(~ended, device=device)
            rollout.append((features, pairs, ranks, distribution.log_prob(ranks), torch.as_tensor(sample, device=device),
                            value, torch.as_tensor(reward, device=device), alive))
            if args.train_flight:
                flight_reward = (.5 * (shaped_reward + reward[:, None])).astype(np.float32)
                flight_rollout.append((normalized["obs"].reshape(*swarm_shape, -1), actions.reshape(*swarm_shape, -1),
                                       flight_outputs["log_prob_actions"].reshape(swarm_shape),
                                       flight_outputs["values"].reshape(swarm_shape),
                                       torch.as_tensor(flight_reward, device=device), alive[:, None]))
            episodes += int(ended.sum())
            total_steps += steps_per_tick

        with torch.no_grad():
            next_distribution, next_value = policy(*observation[:2])
            if args.train_flight:
                observations, pos, vel, goals, _ = state
                _, _, active, sample = observation
                offsets = controller.preview_offsets(next_distribution.sample().cpu().numpy(), sample, active,
                                                     pos, vel, goals)
                next_flight_value = flight(flight_input(observations, offsets), recurrent,
                                           values_only=True)["values"].reshape(swarm_shape)
        check_joint_update = args.train_flight and not verified_joint_update
        if check_joint_update:
            flight_parameter = next(flight.action_parameterization.parameters())
            initial_flight = flight_parameter.detach().clone()
            initial_priority = policy.priority.weight.detach().clone()
        loss = optimize(policy, optimizer, rollout, next_value)
        flight_loss = None
        if flight_optimizer is not None:
            flight_loss = optimize_flight(flight, flight_optimizer, flight_rollout, next_flight_value, cfg, rnn_size)
            if not np.isfinite(flight_loss):
                raise FloatingPointError("Nonfinite flight PPO loss")
        if check_joint_update:
            flight_delta = float((flight_parameter.detach() - initial_flight).abs().max())
            priority_delta = float((policy.priority.weight.detach() - initial_priority).abs().max())
            verified_joint_update = flight_delta > 0 and priority_delta > 0
            print(json.dumps(dict(event="joint_update_check", steps=total_steps,
                                  flight_weight_delta=flight_delta, priority_weight_delta=priority_delta,
                                  flight_loss=flight_loss)), flush=True)
        elapsed = time.perf_counter() - begun
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite PPO loss")
        fps = ticks * steps_per_tick / elapsed
        if warmed_up:
            rates.append(fps)
        update += 1
        total_decisions += sampled
        if args.train_steps is not None:
            if update == initial_update + 1 or total_steps >= next_save or total_steps >= next_milestone:
                checkpoint(latest)
                next_save = (total_steps // args.checkpoint_steps + 1) * args.checkpoint_steps
            if total_steps >= next_milestone:
                checkpoint(args.output / f"checkpoint_{total_steps:012d}.pt")
                next_milestone = (total_steps // args.milestone_steps + 1) * args.milestone_steps
        metrics = dict(update=update, steps=total_steps, fps=round(fps), episodes=episodes,
                       loss=round(loss, 5), decisions=sampled, arrivals=arrivals,
                       hold_fraction=round(waits / (ticks * steps_per_tick), 5),
                       measured=warmed_up, wall_time=time.time())
        if flight_loss is not None:
            metrics["flight_loss"] = round(flight_loss, 5)
        print(json.dumps(metrics), flush=True)
        if args.train_steps is None and rates and time.perf_counter() - measure_begin >= args.seconds:
            break

    elapsed = time.perf_counter() - measure_begin
    summary = dict(mode="pair", device=torch.cuda.get_device_name(), agents=agent_count, arenas=arena_count,
                   pair_limit=controller.limit, horizon=args.horizon, refresh=args.refresh,
                   layouts=[args.layouts[i % len(args.layouts)] for i in range(arena_count)],
                   target_agent_steps=args.train_steps, total_agent_steps=total_steps, seconds=round(elapsed, 3),
                   agent_steps_per_sec=round((total_steps - measure_steps) / elapsed, 1),
                   episodes=episodes, decisions=total_decisions, flight_trained=args.train_flight)
    checkpoint(args.output / "priority.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
