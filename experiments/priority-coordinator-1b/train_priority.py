"""Compare learned coordination policies on top of the same QuadSwarm flight checkpoint."""

import argparse
from contextlib import redirect_stdout
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal

from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.dynamic_priority import DynamicPriority, DynamicPriorityConfig
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.run_layout_suite import make_suite_config, seed_numba
from swarm_rl.evaluate_layouts import load_checkpoint
from swarm_rl.train import register_swarm_components


def run_arena(connection, cfg, seed, report_positions=False):
    np.random.seed(seed)
    seed_numba(seed)
    with connection, open(os.devnull, "w") as silent, redirect_stdout(silent):
        env = make_quadrotor_env_multi(cfg)
        core = env.unwrapped
        core.track_priority_neighbors = True
        for i, drone in enumerate(core.envs):
            drone._seed(seed * 1000 + i)
        obs, _ = env.reset()
        goals = np.asarray([drone.goal for drone in core.envs], dtype=np.float32)

        def snapshot():
            return (
                np.asarray(obs, dtype=np.float32),
                core.pos.astype(np.float32),
                core.vel.astype(np.float32),
                goals,
                core.priority_neighbors,
            )

        connection.send((env.observation_space, env.action_space, core.control_dt,
                         core.collision_threshold, snapshot()))
        while (action := connection.recv()) is not None:
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(np.any(terminated) or np.any(truncated))
            if done:
                goals[:] = [drone.goal for drone in core.envs]
            result = (snapshot(), np.asarray(reward, dtype=np.float32), done)
            if report_positions:
                end_positions = core.layout_terminal_state["positions"] if done else core.pos
                result += (end_positions.astype(np.float32),)
            connection.send(result)
        env.close()


class PriorityPolicy(nn.Module):
    def __init__(self, mode, obs_width, neighbors, message_gain=1.0):
        super().__init__()
        self.mode = mode
        self.message_gain = message_gain
        self.encode = nn.Sequential(nn.Linear(obs_width, 64), nn.Tanh())
        self.bid = nn.Linear(128 if mode == "coordinator" else 64, 1)
        self.decide = nn.Sequential(nn.Linear(66 + neighbors, 64), nn.Tanh(), nn.Linear(64, 1))
        self.value = nn.Linear(128 if mode == "coordinator" else 64, 1)
        nn.init.constant_(self.decide[-1].bias, -1.5)

    def forward(self, features, neighbors, enabled, bid=None, wait=None):
        hidden = self.encode(features)
        if self.mode == "coordinator":
            summary = hidden.mean(dim=-2, keepdim=True).expand_as(hidden)
            hidden = torch.cat((hidden, summary), dim=-1)
        distribution = Normal(self.bid(hidden).squeeze(-1), .5)
        bid = distribution.sample() if bid is None else bid
        log_prob = distribution.log_prob(bid)
        value = self.value(hidden).squeeze(-1)
        if self.mode == "coordinator":
            return bid, torch.zeros_like(bid), log_prob, value

        decision = self.wait_distribution(hidden, bid, neighbors, enabled)
        wait = decision.sample() if wait is None else wait
        return bid, wait, log_prob * enabled + decision.log_prob(wait), value

    def wait_distribution(self, hidden, bid, neighbors, enabled):
        # Neighbor slots are exactly those in the existing local observation.
        neighbors_bid = torch.gather(bid, -1, neighbors.flatten(-2)).reshape_as(neighbors)
        signal = torch.cat((bid.unsqueeze(-1), neighbors_bid), dim=-1)
        received = signal.tanh() * enabled.unsqueeze(-1) * self.message_gain
        flags = enabled.expand_as(bid).unsqueeze(-1)
        return Bernoulli(logits=self.decide(torch.cat((hidden, received, flags), dim=-1)).squeeze(-1))


class GoalControl:
    def __init__(self, mode, initial, control_dt, collision_threshold):
        self.mode = mode
        positions = initial[1]
        self.anchors = np.zeros_like(positions)
        self.remaining = np.zeros(positions.shape[:2], dtype=int)
        self.hold_ticks = max(1, round(.2 / control_dt))
        self.controllers = []
        if mode == "coordinator":
            for row in positions:
                core = SimpleNamespace(
                    pos=row,
                    num_agents=len(row),
                    num_use_neighbor_obs=initial[-1].shape[-1],
                    control_dt=control_dt,
                    collision_threshold=collision_threshold,
                    envs=[SimpleNamespace(obs_repr="xyz_vxyz") for _ in row],
                )
                self.controllers.append(DynamicPriority(core, DynamicPriorityConfig(metric="critic")))

    def offsets(self, state, bid, wait, episode_ended):
        _, pos, vel, goals, neighbors = state
        if self.mode == "coordinator":
            offsets = []
            for i, controller in enumerate(self.controllers):
                controller.core.pos = pos[i]
                controller.core.vel = vel[i]
                controller.core.priority_neighbors = neighbors[i]
                for drone, goal in zip(controller.core.envs, goals[i]):
                    drone.goal = goal
                if episode_ended[i]:
                    controller.remaining.fill(0)
                offsets.append(controller.goal_offsets(-bid[i]))
            return np.asarray(offsets, dtype=np.float32)

        self.remaining[episode_ended] = 0
        starting = (self.remaining == 0) & wait.astype(bool)
        self.anchors[starting] = pos[starting]
        self.remaining[starting] = self.hold_ticks
        active = self.remaining > 0
        self.remaining[active] -= 1
        return np.where(active[..., None], goals - self.anchors, 0.)


def stack_states(states):
    return tuple(np.stack(field) for field in zip(*states))


def save_checkpoint(path, policy, optimizer, args, progress, **extra_state):
    temporary = path.with_suffix(".tmp")
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                    arguments=arguments, progress=progress, **extra_state), temporary)
    temporary.replace(path)


def optimize(policy, optimizer, rollout, last_value, auxiliary=None):
    features, neighbors, enabled, bids, waits, old_log_prob, values, rewards, alive = (
        torch.stack(items) for items in zip(*rollout)
    )
    advantages = torch.empty_like(rewards)
    accumulator = torch.zeros_like(last_value)
    for step in reversed(range(len(values))):
        delta = rewards[step] + .99 * last_value * alive[step] - values[step]
        accumulator = delta + .99 * .95 * alive[step] * accumulator
        advantages[step] = accumulator
        last_value = values[step]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    auxiliary_value = None
    if auxiliary is not None:
        from peer_channel_objective import channel_loss

        supervision, weight = auxiliary
        conflicts, priorities = (torch.stack(items) for items in zip(*supervision))
    for _ in range(2):
        for batch in torch.randperm(len(values), device=values.device).split(16):
            _, _, log_prob, value = policy(features[batch], neighbors[batch], enabled[batch],
                                           bids[batch], waits[batch])
            ratio = (log_prob - old_log_prob[batch]).clamp(-20, 20).exp()
            unclipped = ratio * advantages[batch]
            clipped = ratio.clamp(.8, 1.2) * advantages[batch]
            loss = -torch.minimum(unclipped, clipped).mean() + .5 * (value - returns[batch]).square().mean()
            if auxiliary is not None:
                auxiliary_value = channel_loss(policy, features[batch], neighbors[batch], enabled[batch],
                                               conflicts[batch], priorities[batch])
                loss = loss + weight * auxiliary_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
    return float(loss.detach()), None if auxiliary_value is None else float(auxiliary_value.detach())


def train(args, connections, first, cfg, resume):
    progress = resume["progress"] if resume else {}
    session_seed = args.seed + progress.get("updates", 0)
    torch.manual_seed(session_seed)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    obs_space, action_space, control_dt, threshold, _ = first[0]
    state = stack_states([item[-1] for item in first])
    flight = create_actor_critic(cfg, gym.spaces.Dict({"obs": obs_space}), action_space)
    flight.load_state_dict(load_checkpoint(args.flight_checkpoint)["model"], strict=True)
    flight.eval().to(device).requires_grad_(False)
    swarm_shape = state[0].shape[:2]
    recurrent = torch.zeros(int(np.prod(swarm_shape)), get_rnn_size(cfg), device=device)
    control = GoalControl(args.mode, state, control_dt, threshold)
    features_width = 9 if args.mode == "coordinator" else state[0].shape[-1]
    policy = PriorityPolicy(args.mode, features_width, state[-1].shape[-1], args.peer_message_gain).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    if resume:
        policy.load_state_dict(resume["policy"])
        optimizer.load_state_dict(resume["optimizer"])
    rng = np.random.default_rng(session_seed)
    done = np.ones(args.arenas, dtype=bool)
    channel = rng.random(args.arenas) >= 1 / 3
    if args.arenas > 1:
        channel[:2] = [True, False]
    channel_ticks = progress.get("channel_ticks", 0)
    masked_ticks = progress.get("masked_ticks", 0)
    episodes = progress.get("episodes", 0)
    total_steps = progress.get("steps", 0)
    update = progress.get("updates", 0)
    initial_update = update
    latest = args.output / "latest.pt"
    next_save = (total_steps // args.checkpoint_steps + 1) * args.checkpoint_steps
    next_milestone = (total_steps // args.milestone_steps + 1) * args.milestone_steps
    swarm_size = int(np.prod(swarm_shape))
    rates = []
    if args.peer_use_loss > 0:
        from peer_channel_objective import conflict_supervision
    measured_start = time.perf_counter()
    steps_at_start = total_steps

    def policy_input():
        obs, pos, vel, goals, neighbors = state
        features = (np.concatenate((pos / 5, vel, (goals - pos) / 5), axis=-1)
                    if args.mode == "coordinator" else obs)
        return (torch.as_tensor(features, device=device),
                torch.as_tensor(neighbors, device=device),
                torch.as_tensor(channel[:, None], dtype=torch.float32, device=device))

    def training_progress():
        return dict(steps=total_steps, updates=update, episodes=episodes,
                    channel_ticks=channel_ticks, masked_ticks=masked_ticks)

    while args.train_steps is None or total_steps < args.train_steps:
        warmed_up = update - initial_update >= args.warmup_updates
        if update - initial_update == args.warmup_updates:
            torch.cuda.synchronize()
            measured_start = time.perf_counter()
            steps_at_start = total_steps
        begun = time.perf_counter()
        rollout = []
        supervision = []
        ticks = args.rollout
        if args.train_steps is not None:
            remaining_ticks = (args.train_steps - total_steps + swarm_size - 1) // swarm_size
            ticks = min(ticks, remaining_ticks)
        for _ in range(ticks):
            with torch.no_grad():
                inputs = policy_input()
                if args.peer_use_loss > 0:
                    supervision.append(conflict_supervision(state, device))
                bids, waits, log_prob, value = policy(*inputs)
                offsets = control.offsets(state, bids.cpu().numpy(), waits.cpu().numpy(), done)
                observation = torch.as_tensor(state[0], device=device).clone()
                observation[:, :, :3] += torch.as_tensor(offsets, device=device)
                normalized = flight.normalize_obs({"obs": observation.flatten(0, 1)})
                flight(normalized, recurrent)
                commands = flight.action_distribution().means.cpu().numpy().reshape(*swarm_shape, -1)
                commands = np.clip(commands, action_space.low, action_space.high)
            for connection, command in zip(connections, commands):
                connection.send(command)
            results = [connection.recv() for connection in connections]
            state = stack_states([result[0] for result in results])
            reward = np.asarray([result[1] for result in results])
            team_reward = reward.mean(axis=-1, keepdims=True)
            reward = (np.broadcast_to(team_reward, reward.shape).copy() if args.mode == "coordinator"
                      else (reward + team_reward) * .5)
            done = np.asarray([result[2] for result in results])
            alive = torch.as_tensor(~done[:, None], device=device)
            rollout.append((*inputs, bids, waits, log_prob, value, torch.as_tensor(reward, device=device), alive))
            episodes += int(done.sum())
            if args.mode == "peer":
                channel_ticks += int(channel.sum())
                masked_ticks += int((~channel).sum())
                channel[done] = rng.random(int(done.sum())) >= 1 / 3
            total_steps += swarm_size
        with torch.no_grad():
            last_value = policy(*policy_input())[-1]
        auxiliary = (supervision, args.peer_use_loss) if supervision else None
        loss, auxiliary_loss = optimize(policy, optimizer, rollout, last_value, auxiliary)
        torch.cuda.synchronize()
        wall = time.perf_counter() - begun
        steps_per_update = ticks * swarm_size
        if warmed_up:
            rates.append(steps_per_update / wall)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite PPO loss")
        update += 1
        if args.train_steps is not None:
            if update == initial_update + 1 or total_steps >= next_save or total_steps >= next_milestone:
                save_checkpoint(latest, policy, optimizer, args, training_progress())
                next_save = (total_steps // args.checkpoint_steps + 1) * args.checkpoint_steps
            if total_steps >= next_milestone:
                milestone = args.output / f"checkpoint_{total_steps:012d}.pt"
                save_checkpoint(milestone, policy, optimizer, args, training_progress())
                next_milestone = (total_steps // args.milestone_steps + 1) * args.milestone_steps
        metrics = dict(update=update, steps=total_steps, fps=round(steps_per_update / wall),
                       episodes=episodes, mean_loss=round(loss, 5), measured=warmed_up,
                       wall_time=time.time())
        if auxiliary_loss is not None:
            metrics["peer_auxiliary_loss"] = round(auxiliary_loss, 5)
        print(json.dumps(metrics), flush=True)
        if args.train_steps is None and rates and time.perf_counter() - measured_start >= args.seconds:
            break
    wall = time.perf_counter() - measured_start
    measured_steps = total_steps - steps_at_start
    summary = dict(
        mode=args.mode, device=torch.cuda.get_device_name(), agents=args.agents, arenas=args.arenas,
        layouts=[args.layouts[i % len(args.layouts)] for i in range(args.arenas)],
        rollout=args.rollout, warmup_updates=args.warmup_updates, measured_updates=len(rates),
        total_agent_steps=total_steps, target_agent_steps=args.train_steps,
        agent_steps=measured_steps, seconds=round(wall, 3), agent_steps_per_sec=round(measured_steps / wall, 1),
        scene_steps_per_sec=round(measured_steps / wall / args.agents, 1),
        per_update_fps=[round(rate, 1) for rate in rates],
        episodes=episodes, channel_arena_steps=channel_ticks, masked_arena_steps=masked_ticks,
        peak_cuda_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
    )
    save_checkpoint(args.output / "priority.pt", policy, optimizer, args, training_progress())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("coordinator", "peer", "pair"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--flight-checkpoint", type=Path)
    parser.add_argument("--train-flight", action="store_true",
                        help="Train the flight policy from scratch alongside the pair coordinator")
    parser.add_argument("--peer-message-gain", type=float, default=1.0)
    parser.add_argument("--peer-use-loss", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--arenas", type=int, default=6)
    parser.add_argument("--layouts", nargs="+", default=["narrow_open", "narrow_deadend"])
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--checkpoint-steps", type=int, default=5_000_000)
    parser.add_argument("--milestone-steps", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=.25)
    parser.add_argument("--refresh", type=float, default=.10)
    args = parser.parse_args()
    if args.train_flight and (args.mode != "pair" or args.flight_checkpoint):
        parser.error("--train-flight requires pair mode and starts from random flight weights")
    if not args.train_flight and not args.flight_checkpoint:
        parser.error("--flight-checkpoint is required when the flight policy is frozen")
    if not np.isfinite(args.peer_message_gain) or args.peer_message_gain <= 0:
        parser.error("--peer-message-gain must be finite and positive")
    if not np.isfinite(args.peer_use_loss) or args.peer_use_loss < 0:
        parser.error("--peer-use-loss must be finite and nonnegative")
    if args.mode != "peer" and (args.peer_message_gain != 1.0 or args.peer_use_loss != 0.0):
        parser.error("The message settings require peer mode")
    if args.train_steps is not None and min(args.train_steps, args.checkpoint_steps, args.milestone_steps) < 1:
        parser.error("Training and checkpoint step counts must be positive")
    if args.mode == "pair" and (args.agents < 2 or min(args.pairs, args.horizon, args.refresh) <= 0):
        parser.error("Pair coordination needs at least two agents and positive pair settings")
    args.output.mkdir(parents=True, exist_ok=args.train_steps is not None)
    latest = args.output / "latest.pt"
    resume = (torch.load(latest, map_location="cpu", weights_only=True)
              if args.train_steps is not None and latest.is_file() else None)
    if resume:
        if resume["arguments"].get("train_flight", False) != args.train_flight:
            raise ValueError("Cannot change whether flight is trained on resume")
        if (resume["arguments"].get("peer_message_gain", 1.0), resume["arguments"].get("peer_use_loss", 0.0)) != (
            args.peer_message_gain, args.peer_use_loss
        ):
            raise ValueError("Cannot change the message objective on resume")
        settings = ("mode", "agents", "layouts", "rollout", "seed")
        if args.mode == "pair":
            settings += ("pairs", "horizon", "refresh")
        else:
            settings += ("arenas",)
        for name in settings:
            if resume["arguments"][name] != getattr(args, name):
                raise ValueError(f"Cannot resume with a different {name}")
        previous_arenas = resume["arguments"]["arenas"]
        if previous_arenas != args.arenas:
            print(json.dumps(dict(event="resume_arena_change", steps=resume["progress"]["steps"],
                                  previous_arenas=previous_arenas, arenas=args.arenas)), flush=True)
    register_swarm_components()
    config = json.loads(args.config.read_text())
    cfg = make_suite_config(config, args.layouts[0], args.agents, 15.)
    cfg.device = "gpu"
    if cfg.use_rnn or cfg.quads_neighbor_visible_num < 1:
        raise ValueError("This benchmark expects the feed-forward swarm checkpoint with local neighbors")
    children, connections = [], []
    context = mp.get_context("spawn")
    try:
        for i in range(args.arenas):
            parent, child = context.Pipe()
            arena_cfg = make_suite_config(config, args.layouts[i % len(args.layouts)], args.agents, 15.)
            previous_updates = resume["progress"]["updates"] if resume else 0
            arena_cfg.layout_seed = args.seed + 7919 * i + args.arenas * previous_updates
            arena_cfg.quads_static_priority = "off"
            process = context.Process(target=run_arena,
                                      args=(child, arena_cfg, arena_cfg.layout_seed, args.mode == "pair"))
            process.start()
            child.close()
            children.append(process)
            connections.append(parent)
        first = [connection.recv() for connection in connections]
        if args.mode == "pair":
            from train_pair_priority import train_pair

            train_pair(args, connections, first, cfg, resume, save_checkpoint, stack_states)
        else:
            train(args, connections, first, cfg, resume)
    finally:
        for connection in connections:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            connection.close()
        for process in children:
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()
