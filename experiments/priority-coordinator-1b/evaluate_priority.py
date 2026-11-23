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
from train_priority import PriorityPolicy


MODES = ("baseline", "static_brake", "coordinator")
AGENT_COUNTS = (8, 100)
SEED = 50000
DURATION = 15.
_FLIGHT_STATE = None
_RANKER = None
_CONFIG = None


def initialize(config, flight_path, coordinator_path):
    global _CONFIG, _FLIGHT_STATE, _RANKER
    torch.set_num_threads(1)
    register_swarm_components()
    _CONFIG = json.loads(config.read_text())
    _FLIGHT_STATE = load_checkpoint(flight_path)["model"]
    checkpoint = torch.load(coordinator_path, map_location="cpu", weights_only=True)
    if checkpoint["arguments"]["mode"] != "coordinator" or checkpoint["progress"]["steps"] < 1_000_000_000:
        raise ValueError("Expected the completed learned coordinator checkpoint")
    _RANKER = PriorityPolicy("coordinator", 9, _CONFIG["quads_neighbor_visible_num"])
    _RANKER.load_state_dict(checkpoint["policy"], strict=True)
    _RANKER.eval()


class LearnedPriority(DynamicPriority):
    def _priority(self, distance, critic_values):
        pos = self.core.pos.astype(np.float32)
        vel = self.core.vel.astype(np.float32)
        goals = np.asarray([drone.goal for drone in self.core.envs], dtype=np.float32)
        features = torch.from_numpy(np.concatenate((pos / 5, vel, (goals - pos) / 5), axis=-1))
        hidden = _RANKER.encode(features)
        pooled = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
        return _RANKER.bid(torch.cat((hidden, pooled), dim=-1)).squeeze(-1).numpy()


def record_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(value, indent=2) + "\n")
    staged.replace(path)


def episode(agents, layout, mode, output):
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
            coordinator = LearnedPriority(core, DynamicPriorityConfig()) if mode == "coordinator" else None
            brake = PriorityBrake(core, BrakeConfig()) if mode == "static_brake" else None
            obs, _ = env.reset()
            plan = core.layout_plan
            model = create_actor_critic(cfg, gym.spaces.Dict({"obs": env.observation_space}), env.action_space)
            model.load_state_dict(_FLIGHT_STATE, strict=True)
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


def summarize(rows, output, layouts):
    keys = {(n, layout["id"], mode) for n in AGENT_COUNTS for layout in layouts for mode in MODES}
    index = {(r["agents"], r["layout_id"], r["mode"]): r for r in rows}
    if len(rows) != len(index) or index.keys() != keys:
        raise RuntimeError("The comparison is missing scenes or has duplicate results")
    for n in AGENT_COUNTS:
        for layout in layouts:
            if len({index[n, layout["id"], mode]["layout_sha256"] for mode in MODES}) != 1:
                raise ValueError(f"Scenario differs across modes for {n} / {layout['id']}")
    metrics = ("success_rate", "collision_rate", "neighbor_collision_rate", "obstacle_collision_rate",
               "timeout_rate", "stalled_rate", "hold_fraction")
    summary = []
    for n in AGENT_COUNTS:
        for kind in ("corridor", "dead_end", "all"):
            for mode in MODES:
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
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    layouts = [layout for layout in catalogue() if layout["paired_with"] is not None]
    if len(layouts) != 16:
        raise RuntimeError("Expected eight paired corridor/dead-end layouts")
    manifest = dict(
        agents=list(AGENT_COUNTS), modes=list(MODES), seed=SEED, duration=DURATION,
        layouts=[layout["id"] for layout in layouts], static_brake=vars(BrakeConfig()),
        files={file.name: hashlib.sha256(file.read_bytes()).hexdigest()
               for file in (args.config, args.flight, args.coordinator, Path(__file__), Path(__file__).with_name("priority_brake.py"))},
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to merge results from different configurations")
    record_json(manifest_path, manifest)
    tasks = [(n, layout, mode, args.output) for n in reversed(AGENT_COUNTS) for layout in layouts for mode in MODES]
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=initialize, initargs=(args.config, args.flight, args.coordinator)
    ) as pool:
        futures = [pool.submit(episode, *task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} N={row['agents']} {row['layout_id']} {row['mode']} "
                  f"success={row['success_rate']:.3f} wall={row['wall_seconds']:.1f}s", flush=True)
    summarize(rows, args.output, layouts)
    print("96 paired-mode episodes complete. Scene geometry verified across all modes.", flush=True)


if __name__ == "__main__":
    main()
