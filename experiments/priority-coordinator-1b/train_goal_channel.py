"""Fine-tune gated waiting decisions against verified collision-free arrivals."""

import argparse
from contextlib import redirect_stdout
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import gymnasium as gym
from numba import njit
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.distributions import Bernoulli

from gym_art.quadrotor_multi.obstacles.layouts import obstacle_clearance
from priority_brake import BrakeConfig, PriorityBrake, predicts_conflict
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.evaluate_layouts import load_checkpoint
from swarm_rl.run_layout_suite import make_suite_config, seed_numba
from swarm_rl.train import register_swarm_components
from train_pair_priority import advantage_targets
from train_priority import PriorityPolicy, save_checkpoint, stack_states


AGENTS = 100
LAYOUTS = ("narrow_open", "narrow_deadend", "straight_open", "elbow_open", "zigzag_open",
           "funnel_open", "fork_open", "wide_u_open", "comb_open")
GAMMA = .999
GAE_LAMBDA = .995
BONUS = 5.
MESSAGE_PROBABILITY = 1 / 3
MILESTONE = 25_000_000
SAVE_INTERVAL = 1_000_000
SEED = 1729


class ArrivalReward:
    def __init__(self, core):
        self.contact = np.zeros(core.num_agents, dtype=bool)
        self.arrived = self.contact.copy()
        self.safe = self.contact.copy()
        self.near_ticks = np.zeros(core.num_agents, dtype=int)
        self.hold_ticks = max(1, round(.5 / core.control_dt))
        self.collision_threshold = np.nextafter(core.collision_threshold, np.inf)
        self.start(core)
        self.update(core.pos, np.zeros_like(self.contact))

    def start(self, core):
        self.contact.fill(False)
        self.arrived.fill(False)
        self.safe.fill(False)
        self.near_ticks.fill(0)
        self.plan = core.layout_plan

    def update(self, positions, room_contacts):
        pairs = cKDTree(positions).query_pairs(self.collision_threshold, output_type="ndarray")
        self.contact[pairs.ravel()] = True
        self.contact |= room_contacts
        clean = np.flatnonzero(~self.contact)
        clearance = obstacle_clearance(positions[clean, :2], self.plan.obstacles, self.plan.spec.obstacle_size / 2)
        self.contact[clean[clearance <= self.plan.spec.agent_radius]] = True
        near = np.linalg.norm(positions - self.plan.goals, axis=-1) <= .3
        self.near_ticks = np.where(near, self.near_ticks + 1, 0)
        self.arrived |= self.near_ticks >= self.hold_ticks
        safe = self.arrived & ~self.contact
        delta = safe.astype(np.int8) - self.safe
        self.safe = safe
        return delta


def run_arena(connection, cfg, seed):
    random.seed(seed)
    np.random.seed(seed)
    seed_numba(seed)
    with connection, open(os.devnull, "w") as silent, redirect_stdout(silent):
        env = make_quadrotor_env_multi(cfg)
        try:
            core = env.unwrapped
            core.track_priority_neighbors = True
            obs, _ = env.reset()
            reward = ArrivalReward(core)

            def snapshot():
                return (np.asarray(obs, dtype=np.float32), core.pos.copy(), core.vel.copy(),
                        core.layout_plan.goals, core.priority_neighbors)

            connection.send((env.observation_space, env.action_space, core.control_dt,
                             core.collision_threshold, snapshot()))
            while (actions := connection.recv()) is not None:
                obs, sim_reward, ended, truncated, _ = env.step(actions)
                done = bool(np.any(ended) or np.any(truncated))
                if done:
                    terminal = core.layout_terminal_state
                    pos, room = terminal["positions"], terminal["room_contacts"]
                else:
                    pos = core.pos
                    room = np.array([d.dynamics.crashed_floor or d.dynamics.crashed_wall
                                     or d.dynamics.crashed_ceiling for d in core.envs], dtype=bool)
                delta = reward.update(pos, room)
                final = int(reward.safe.sum()) if done else 0
                if done:
                    reward.start(core)
                connection.send((snapshot(), np.asarray(sim_reward, dtype=np.float32), done, delta, final))
        finally:
            env.close()


@njit(cache=True)
def resolve_with_opportunities(nominal, stopped, order, threshold_squared, requests):
    """Same realized braking as the filter, plus potential decisions on this step."""
    accepted = nominal.copy()
    braking = np.zeros(len(order), dtype=np.bool_)
    conflict = braking.copy()
    for rank, drone in enumerate(order):
        for senior in order[:rank]:
            if predicts_conflict(nominal, accepted, drone, senior, threshold_squared):
                conflict[drone] = True
                if requests[drone]:
                    braking[drone] = True
                    accepted[drone] = stopped[drone]
                break
    return braking, conflict


class TrainingBrake(PriorityBrake):
    def __init__(self, control_dt, threshold):
        config = BrakeConfig()
        core = SimpleNamespace(num_agents=AGENTS, pos=np.zeros((AGENTS, 3)),
                               control_dt=control_dt, collision_threshold=threshold,
                               envs=[SimpleNamespace(obs_repr="xyz_vxyz")])
        super().__init__(core, config)
        self.horizon = config.horizon
        self.intent_times = self.nominal_times - self.brake_times
        self.refresh_ticks = max(1, round(.1 / control_dt))
        self.reset()

    def reset(self):
        self.tick = 0
        self.order = self.ids
        self.remaining.fill(0)
        self.waited.fill(0)
        self.promotion_left = 0
        self.promoted = -1

    def priority_order(self):
        return self.order

    def apply(self, positions, velocities, goals, bids, requests):
        if self.tick % self.refresh_ticks == 0:
            self.order = np.lexsort((self.ids, -bids))
        self.tick += 1
        delta = goals - positions
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
        intent = delta / np.maximum(distance, 1e-6)
        intent *= np.minimum(distance / self.horizon, .8)
        intent *= np.maximum(1. - speed / .8, 0.)
        nominal = (positions[:, None, :] + velocities[:, None, :] * self.nominal_times
                   + intent[:, None, :] * self.intent_times)
        stopped = positions[:, None, :] + velocities[:, None, :] * self.brake_times
        braking, conflict = resolve_with_opportunities(
            nominal, stopped, self.order_with_promotion(), self.threshold_squared, requests)
        started = braking & (self.waited == 0)
        self.anchors[started] = positions[started]
        self.waited = np.where(braking, self.waited + 1, 0)
        return np.where(braking[:, None], goals - self.anchors, 0.), conflict, braking


def inputs(policy, state, enabled, device):
    obs, _, _, _, neighbor_ids = state
    hidden = policy.encode(torch.as_tensor(obs, device=device))
    bids = policy.bid(hidden).squeeze(-1)
    ids = torch.as_tensor(neighbor_ids, dtype=torch.long, device=device)
    received = bids.gather(-1, ids.flatten(-2)).reshape_as(ids)
    mask = torch.as_tensor(enabled, device=device)[:, None, None]
    signals = torch.cat((bids.unsqueeze(-1), received * mask), dim=-1).tanh() * policy.message_gain
    action_input = torch.cat((hidden, signals, torch.ones_like(bids.unsqueeze(-1))), dim=-1)
    return bids, hidden, action_input


def optimize(policy, optimizer, rollout, next_value):
    action_inputs, actions, old_log_prob, values, rewards, alive, opportunities = (
        torch.stack(items) for items in zip(*rollout)
    )
    returns, advantages = advantage_targets(rewards, values, alive, next_value, GAMMA, GAE_LAMBDA)
    action_inputs = action_inputs.flatten(0, 2)
    actions, old_log_prob, opportunities = (x.flatten() for x in (actions, old_log_prob, opportunities))
    for _ in range(2):
        for batch in torch.randperm(len(returns), device=returns.device).split(8192):
            features = action_inputs[batch]
            distr = Bernoulli(logits=policy.decide(features).squeeze(-1))
            ratio = (distr.log_prob(actions[batch]) - old_log_prob[batch]).clamp(-20, 20).exp()
            adv = advantages[batch]
            clipped = torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv)
            mask = opportunities[batch]
            actor_loss = -(clipped * mask).sum() / mask.sum().clamp_min(1)
            value = policy.value(features[:, :64]).squeeze(-1)
            loss = actor_loss + .5 * (value - returns[batch]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((*policy.decide.parameters(), *policy.value.parameters()), .5)
            optimizer.step()
    return float(loss.detach())


def train(args, cfg, connections, first, start, resume):
    progress = start["progress"]
    steps, update, episodes = (progress[k] for k in ("steps", "updates", "episodes"))
    torch.manual_seed(SEED + update)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    obs_space, action_space, dt, threshold, _ = first[0]
    state = stack_states([item[-1] for item in first])
    per_tick = args.arenas * AGENTS
    rng = np.random.default_rng(SEED + update)
    enabled = rng.random(args.arenas) < MESSAGE_PROBABILITY
    gate = [TrainingBrake(dt, threshold) for _ in first]

    flight = create_actor_critic(cfg, gym.spaces.Dict({"obs": obs_space}), action_space)
    flight.load_state_dict(load_checkpoint(args.flight)["model"], strict=True)
    flight.eval().to(device).requires_grad_(False)
    rnn = torch.zeros(per_tick, get_rnn_size(cfg), device=device)
    policy = PriorityPolicy("peer", state[0].shape[-1], state[-1].shape[-1],
                            start["arguments"]["peer_message_gain"]).to(device)
    policy.load_state_dict(start["policy"], strict=True)
    policy.encode.requires_grad_(False)
    policy.bid.requires_grad_(False)
    optimizer = torch.optim.Adam((*policy.decide.parameters(), *policy.value.parameters()), lr=1e-4)
    if resume:
        optimizer.load_state_dict(start["optimizer"])

    target = args.train_steps
    next_save = (steps // SAVE_INTERVAL + 1) * SAVE_INTERVAL
    while steps < target:
        begun = time.perf_counter()
        rollout = []
        holds = opportunities = finished = 0
        safe_by_kind = {name: [0, 0] for name in ("corridor", "dead_end")}
        ticks = min(128, (target - steps + per_tick - 1) // per_tick)
        boundary = (steps // MILESTONE + 1) * MILESTONE
        ticks = min(ticks, (boundary - steps + per_tick - 1) // per_tick)
        for _ in range(ticks):
            obs, pos, vel, goals, _ = state
            with torch.no_grad():
                bids, hidden, action_inputs = inputs(policy, state, enabled, device)
                distr = Bernoulli(logits=policy.decide(action_inputs).squeeze(-1))
                requests = distr.sample()
                values = policy.value(hidden).squeeze(-1)
                scores, requested = bids.cpu().numpy(), requests.cpu().numpy().astype(bool)
                offsets, conflicts, braking = stack_states([
                    control.apply(pos[i], vel[i], goals[i], scores[i], requested[i])
                    for i, control in enumerate(gate)
                ])
                holds += int(braking.sum())
                opportunities += int(conflicts.sum())
                observations = torch.as_tensor(obs, device=device).clone()
                observations[:, :, :3] += torch.as_tensor(offsets, dtype=observations.dtype, device=device)
                normalized = flight.normalize_obs({"obs": observations.flatten(0, 1)})
                flight(normalized, rnn)
                actions = flight.action_distribution().means.cpu().numpy().reshape(args.arenas, AGENTS, -1)
            for connection, action in zip(connections, np.clip(actions, action_space.low, action_space.high)):
                connection.send(action)
            results = [connection.recv() for connection in connections]
            state = stack_states([result[0] for result in results])
            rewards = np.stack([result[1] + BONUS * result[3] for result in results]).astype(np.float32)
            rewards = .5 * (rewards + rewards.mean(axis=1, keepdims=True))
            done = np.asarray([result[2] for result in results])
            rollout.append((action_inputs, requests, distr.log_prob(requests), values,
                            torch.as_tensor(rewards, device=device),
                            torch.as_tensor(~done[:, None], device=device),
                            torch.as_tensor(conflicts, device=device)))
            for i in np.flatnonzero(done):
                gate[i].reset()
                kind = "dead_end" if LAYOUTS[i % len(LAYOUTS)].endswith("deadend") else "corridor"
                safe_by_kind[kind][0] += results[i][4]
                safe_by_kind[kind][1] += 1
            enabled[done] = rng.random(int(done.sum())) < MESSAGE_PROBABILITY
            finished += int(done.sum())
            steps += per_tick
        with torch.no_grad():
            _, hidden, _ = inputs(policy, state, enabled, device)
            next_value = policy.value(hidden).squeeze(-1)
        loss = optimize(policy, optimizer, rollout, next_value)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite channel PPO loss")
        update += 1
        episodes += finished
        progress = dict(steps=steps, updates=update, episodes=episodes)
        if steps >= next_save or steps == target or update == start["progress"]["updates"] + 1:
            save_checkpoint(args.output / "latest.pt", policy, optimizer, args, progress)
            next_save = (steps // SAVE_INTERVAL + 1) * SAVE_INTERVAL
        if steps % MILESTONE == 0 or steps == target:
            save_checkpoint(args.output / f"checkpoint_{steps:012d}.pt", policy, optimizer, args, progress)
        print(json.dumps(dict(update=update, steps=steps, episodes=episodes, loss=round(loss, 5),
                              fps=round(ticks * per_tick / (time.perf_counter() - begun)),
                              hold_fraction=holds / (ticks * per_tick),
                              opportunity_fraction=opportunities / (ticks * per_tick),
                              safe_arrivals=safe_by_kind)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "flight", "peer", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--arenas", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=200_000_000)
    args = parser.parse_args()
    args.mode = "peer"
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    resume = latest.is_file()
    start = torch.load(latest if resume else args.peer, map_location="cpu", weights_only=True)
    if start["arguments"]["mode"] != "peer" or start["progress"]["steps"] < 100_000_000:
        raise ValueError("Expected the trained weighted channel")
    args.peer_message_gain = start["arguments"]["peer_message_gain"]
    if resume and start["arguments"]["arenas"] != args.arenas:
        raise ValueError("Cannot change parallel environments on resume")
    if args.train_steps % (AGENTS * args.arenas) or MILESTONE % (AGENTS * args.arenas):
        raise ValueError("Choose a run length and worker count which land exactly on milestones")
    print(json.dumps(dict(event="channel_start", device=torch.cuda.get_device_name(),
                          steps=start["progress"]["steps"], target=args.train_steps, agents=AGENTS,
                          arenas=args.arenas, layouts=LAYOUTS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                          safe_arrival_bonus=BONUS, neighbor_message_probability=MESSAGE_PROBABILITY)), flush=True)
    register_swarm_components()
    original = json.loads(args.config.read_text())
    cfg = make_suite_config(original, LAYOUTS[0], AGENTS, 15.)
    cfg.device = "gpu"
    children, connections = [], []
    context = mp.get_context("spawn")
    try:
        for i in range(args.arenas):
            parent, child = context.Pipe()
            scene = make_suite_config(original, LAYOUTS[i % len(LAYOUTS)], AGENTS, 15.)
            seed = 1_000_000 + SEED + 7919 * i + start["progress"]["updates"] * args.arenas
            scene.layout_seed = seed
            scene.quads_static_priority = "off"
            process = context.Process(target=run_arena, args=(child, scene, seed))
            process.start()
            child.close()
            children.append(process)
            connections.append(parent)
        first = [connection.recv() for connection in connections]
        train(args, cfg, connections, first, start, resume)
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
