"""Paired QuadSwarm comparison of baseline, static braking, and learned coordination."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
from statistics import mean
import time

import gymnasium as gym
import numpy as np
import torch

from gym_art.quadrotor_multi.obstacles.layouts import obstacle_clearance
from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue
from priority_brake import BrakeConfig, PriorityBrake
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.dynamic_priority import DynamicPriority, DynamicPriorityConfig
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.evaluate_layouts import load_checkpoint
from swarm_rl.layout_metrics import LayoutMetrics
from swarm_rl.run_layout_suite import make_suite_config, seed_numba
from swarm_rl.train import register_swarm_components
from train_priority import GoalControl, PriorityPolicy


MODES = ("baseline", "static_brake", "coordinator")
AGENT_COUNTS = (8, 100)
SEED = 50000
DURATION = 15.
_FLIGHT_STATE = None
_RANKER = None
_PEER_CHECKPOINT = None
_CONFIG = None


def initialize(config, flight_path, coordinator_path, peer_path=None):
    global _CONFIG, _FLIGHT_STATE, _RANKER, _PEER_CHECKPOINT
    torch.set_num_threads(1)
    register_swarm_components()
    _CONFIG = json.loads(config.read_text())
    _FLIGHT_STATE = load_checkpoint(flight_path)["model"]
    if coordinator_path:
        checkpoint = torch.load(coordinator_path, map_location="cpu", weights_only=True)
        if checkpoint["arguments"]["mode"] != "coordinator" or checkpoint["progress"]["steps"] < 1_000_000_000:
            raise ValueError("Expected the completed learned coordinator checkpoint")
        _RANKER = PriorityPolicy("coordinator", 9, _CONFIG["quads_neighbor_visible_num"])
        _RANKER.load_state_dict(checkpoint["policy"], strict=True)
        _RANKER.eval()
    if peer_path:
        _PEER_CHECKPOINT = torch.load(peer_path, map_location="cpu", weights_only=True)
        arguments = _PEER_CHECKPOINT["arguments"]
        target = arguments.get("train_steps") or 1_000_000_000
        if arguments["mode"] != "peer" or _PEER_CHECKPOINT["progress"]["steps"] < target:
            raise ValueError("Expected the completed peer checkpoint")


class LearnedPriority(DynamicPriority):
    def _priority(self, distance, critic_values):
        pos = self.core.pos.astype(np.float32)
        vel = self.core.vel.astype(np.float32)
        goals = np.asarray([drone.goal for drone in self.core.envs], dtype=np.float32)
        features = torch.from_numpy(np.concatenate((pos / 5, vel, (goals - pos) / 5), axis=-1))
        hidden = _RANKER.encode(features)
        pooled = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
        return _RANKER.bid(torch.cat((hidden, pooled), dim=-1)).squeeze(-1).numpy()


class FixedPriority(DynamicPriority):
    def _priority(self, distance, critic_values):
        return -self.ids[:, 0]


class PeerPriority:
    def __init__(self, core, observations, channel_enabled):
        self.core = core
        self.channel_enabled = channel_enabled
        features = torch.as_tensor(observations, dtype=torch.float32)
        gain = _PEER_CHECKPOINT["arguments"].get("peer_message_gain", 1.0)
        self.policy = PriorityPolicy("peer", features.shape[-1], core.num_use_neighbor_obs, gain)
        self.policy.load_state_dict(_PEER_CHECKPOINT["policy"], strict=True)
        self.policy.eval()
        self.goal_control = GoalControl("peer", (None, core.pos[None]), core.control_dt, core.collision_threshold)
        self.no_messages = features.new_zeros((len(features), core.num_use_neighbor_obs + 2))
        self.channel_flag = features.new_ones((len(features), 1))
        self.episode_ended = np.array([False])
        self.yield_ticks = 0

    def policy_inputs(self, model, observations, states):
        features = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        hidden = self.policy.encode(features)
        core = self.core
        channel = self.no_messages
        if self.channel_enabled:
            bid = self.policy.bid(hidden)[:, 0]
            neighbors = torch.as_tensor(core.priority_neighbors, dtype=torch.long)
            signals = torch.cat((bid[:, None], bid[neighbors]), dim=-1).tanh() * self.policy.message_gain
            channel = torch.cat((signals, self.channel_flag), dim=-1)
        logits = self.policy.decide(torch.cat((hidden, channel), dim=-1))
        wait = logits[:, 0].numpy() > 0
        goals = np.asarray([drone.goal for drone in core.envs], dtype=np.float32)
        state = (None, core.pos[None], core.vel[None], goals[None], None)
        self.yield_ticks += int(np.count_nonzero((self.goal_control.remaining[0] > 0) | wait))
        offsets = self.goal_control.offsets(state, None, wait[None], self.episode_ended)[0]
        inputs = features.clone()
        inputs[:, :3] += torch.as_tensor(offsets, dtype=features.dtype)
        return model.normalize_obs({"obs": inputs})


def record_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(value, indent=2) + "\n")
    staged.replace(path)


def episode(agents, layout, mode, output, *, flight_state=None, controller_factory=None):
    layout_id = layout["id"]
    result_path = output / "scenes" / f"n{agents}" / layout_id / f"{mode}.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    cfg = make_suite_config(_CONFIG, layout_id, agents, DURATION)
    cfg.layout_seed = SEED
    cfg.quads_static_priority = "off"
    with open(os.devnull, "w") as silent, redirect_stdout(silent), torch.inference_mode():
        env = make_quadrotor_env_multi(cfg)
        try:
            core = env.unwrapped
            core.capture_terminal_state = True
            random.seed(SEED)
            np.random.seed(SEED)
            seed_numba(SEED)
            torch.manual_seed(SEED)
            for i, drone in enumerate(core.envs):
                drone._seed(1000 * SEED + i)
            ranking = {"coordinator": LearnedPriority, "fixed_rank_local": FixedPriority}.get(mode)
            coordinator = ranking(core, DynamicPriorityConfig()) if ranking else None
            if controller_factory is not None:
                coordinator = controller_factory(core)
            brake = PriorityBrake(core, BrakeConfig()) if mode == "static_brake" else None
            if mode == "peer_on":
                core.track_priority_neighbors = True
            obs, _ = env.reset()
            if mode in ("peer_on", "peer_off"):
                coordinator = PeerPriority(core, np.asarray(obs), channel_enabled=(mode == "peer_on"))
            plan = core.layout_plan
            model = create_actor_critic(cfg, gym.spaces.Dict({"obs": env.observation_space}), env.action_space)
            model.load_state_dict(_FLIGHT_STATE if flight_state is None else flight_state, strict=True)
            model.eval()
            states = torch.zeros(agents, get_rnn_size(cfg))
            metrics = LayoutMetrics(plan, core.control_dt, core.collision_threshold)
            metrics.obstacle_contacts |= obstacle_clearance(
                plan.starts[:, :2], plan.obstacles, plan.spec.obstacle_size / 2
            ) <= plan.spec.agent_radius
            distances = np.linalg.norm(plan.starts[:, None] - plan.starts[None, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            metrics.neighbor_contacts |= np.any(distances <= core.collision_threshold, axis=1)
            begun = time.perf_counter()
            for _ in range(int(np.ceil(DURATION / core.control_dt)) + 2):
                if coordinator is None:
                    observation = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
                    normalized = model.normalize_obs({"obs": observation})
                else:
                    normalized = coordinator.policy_inputs(model, obs, states)
                outputs = model(normalized, states)
                actions = model.action_distribution().means.cpu().numpy().copy()
                next_states = outputs["new_rnn_states"]
                if brake is not None:
                    actions, next_states, _ = brake.actions(model, obs, states, normalized, actions, next_states)
                if not np.isfinite(actions).all():
                    raise FloatingPointError(f"Nonfinite actions in {agents} / {layout_id} / {mode}")
                obs, _, terminated, truncated, _ = env.step(np.clip(actions, env.action_space.low, env.action_space.high))
                states = next_states
                done = bool(np.any(np.logical_or(terminated, truncated)))
                if done:
                    terminal = core.layout_terminal_state
                    metrics.update(terminal["positions"], terminal["room_contacts"])
                    break
                rooms = [drone.dynamics.crashed_floor or drone.dynamics.crashed_wall or drone.dynamics.crashed_ceiling
                         for drone in core.envs]
                metrics.update(core.pos, rooms)
            else:
                raise RuntimeError(f"Episode did not terminate for {agents} / {layout_id} / {mode}")
            walls = json.dumps(plan.to_dict(), sort_keys=True).encode()
            holds = coordinator.yield_ticks if coordinator else brake.brake_ticks if brake else 0
            result = dict(
                agents=agents, layout_id=layout_id, family=layout["family"],
                kind="dead_end" if layout["has_dead_end"] else "corridor",
                mode=mode, seed=SEED, layout_sha256=hashlib.sha256(walls).hexdigest(),
                **metrics.summary(),
                swarm_success=int(all(row["success"] for row in metrics.agent_rows())),
                hold_fraction=holds / (agents * metrics.ticks), wall_seconds=time.perf_counter() - begun,
            )
        finally:
            env.close()
    record_json(result_path, result)
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, output, layouts, modes):
    keys = {(n, layout["id"], mode) for n in AGENT_COUNTS for layout in layouts for mode in modes}
    index = {(r["agents"], r["layout_id"], r["mode"]): r for r in rows}
    if len(rows) != len(index) or index.keys() != keys:
        raise RuntimeError("The comparison is missing scenes or has duplicate results")
    for n in AGENT_COUNTS:
        for layout in layouts:
            if len({index[n, layout["id"], mode]["layout_sha256"] for mode in modes}) != 1:
                raise ValueError(f"Scenario differs across modes for {n} / {layout['id']}")
    metrics = ("success_rate", "collision_rate", "neighbor_collision_rate", "obstacle_collision_rate",
               "timeout_rate", "stalled_rate", "hold_fraction")
    summary = []
    for n in AGENT_COUNTS:
        for kind in ("corridor", "dead_end", "all"):
            for mode in modes:
                group = [r for r in rows if r["agents"] == n and r["mode"] == mode
                         and (kind == "all" or r["kind"] == kind)]
                summary.append(dict(agents=n, kind=kind, mode=mode, scenes=len(group),
                                    **{key: mean(r[key] for r in group) for key in metrics},
                                    full_swarm_successes=sum(r["swarm_success"] for r in group)))
    write_csv(output / "episodes.csv", sorted(rows, key=lambda r: (r["agents"], r["layout_id"], r["mode"])))
    write_csv(output / "summary.csv", summary)
    record_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--flight", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path)
    parser.add_argument("--peer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--modes", nargs="+", choices=(*MODES, "fixed_rank_local", "peer_off", "peer_on"), default=MODES)
    args = parser.parse_args()
    if "coordinator" in args.modes and not args.coordinator:
        parser.error("The coordinator mode requires --coordinator")
    peer_modes = {"peer_off", "peer_on"}.intersection(args.modes)
    if peer_modes and not args.peer:
        parser.error("Peer modes require --peer")
    layouts = [layout for layout in catalogue() if layout["paired_with"] is not None]
    if len(layouts) != 16:
        raise RuntimeError("Expected eight paired corridor/dead-end layouts")
    manifest = dict(
        agents=list(AGENT_COUNTS), modes=list(args.modes), seed=SEED, duration=DURATION,
        layouts=[layout["id"] for layout in layouts], static_brake=vars(BrakeConfig()),
        files={file.name: hashlib.sha256(file.read_bytes()).hexdigest()
               for file in (args.config, args.flight, args.coordinator, args.peer, Path(__file__),
                            Path(__file__).with_name("priority_brake.py")) if file},
    )
    if peer_modes:
        manifest["peer_channel"] = "both" if len(peer_modes) == 2 else next(iter(peer_modes)).removeprefix("peer_")
        trainer = Path(__file__).with_name("train_priority.py")
        manifest["files"][trainer.name] = hashlib.sha256(trainer.read_bytes()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to merge results from different configurations")
    record_json(manifest_path, manifest)
    tasks = [(n, layout, mode, args.output) for n in reversed(AGENT_COUNTS) for layout in layouts for mode in args.modes]
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=initialize, initargs=(args.config, args.flight, args.coordinator, args.peer)
    ) as pool:
        futures = [pool.submit(episode, *task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} N={row['agents']} {row['layout_id']} {row['mode']} "
                  f"success={row['success_rate']:.3f} wall={row['wall_seconds']:.1f}s", flush=True)
    summarize(rows, args.output, layouts, args.modes)
    print(f"{len(tasks)} episodes complete.", flush=True)


if __name__ == "__main__":
    main()
