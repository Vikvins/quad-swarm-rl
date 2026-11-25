"""Compare static and pair-ranked orders using the same all-pairs brake."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import csv
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_pair_checkpoint as pair
import pair_priority
import priority_brake


evaluation = pair.evaluation
BaseMetrics = evaluation.LayoutMetrics
AGENTS = 100
MODES = ("static_brake", "pair_ranked_brake")
CONFIG = priority_brake.BrakeConfig(horizon=.65)


def conflict_groups(first, second, agents):
    parent = list(range(agents))

    def root(agent):
        while parent[agent] != agent:
            agent = parent[agent]
        return agent

    for i, j in zip(first, second):
        parent[root(j)] = root(i)
    groups = {}
    for agent in np.unique(np.r_[first, second]):
        groups.setdefault(root(agent), []).append(agent)
    return groups.values()


class PairRankedBrake(priority_brake.PriorityBrake):
    def __init__(self, core):
        super().__init__(core, CONFIG)
        settings = SimpleNamespace(**(vars(pair._PAIR_SETTINGS) | {"horizon": CONFIG.horizon}))
        self.selector = pair_priority.PairController(1, core.num_agents, core.control_dt,
                                                     core.collision_threshold, settings)
        self.refresh = max(1, round(settings.refresh / core.control_dt))
        self.tick = 0
        self.order = self.ids.copy()
        self.initial = np.ones(1, dtype=bool)
        self.committed = np.zeros(core.num_agents, dtype=bool)
        self.scores = np.zeros(core.num_agents, dtype=np.float32)
        self.rank_changes = 0
        self.rank_inferences = 0

    def priority_order(self):
        if self.tick % self.refresh == 0:
            self.update_order()
        self.tick += 1
        return self.order

    def update_order(self):
        core = self.core
        pos = np.asarray(core.pos, dtype=np.float32)[None]
        vel = np.asarray(core.vel, dtype=np.float32)[None]
        pairs, _, _ = self.selector.observe(pos, vel, self.initial)
        self.initial.fill(False)
        locked = self.selector.locked_edges[0]
        first, second = self.selector.first[locked], self.selector.second[locked]
        active = np.zeros_like(self.committed)
        active[first] = True
        active[second] = True
        new = active & ~self.committed
        if new.any():
            goals = np.asarray([agent.goal for agent in core.envs], dtype=np.float32)[None]
            features = torch.from_numpy(pair_priority.drone_features(pos, vel, goals))
            distribution, _ = pair._PAIR_MODEL(features, torch.from_numpy(pairs))
            self.scores[new] = distribution.mean.numpy()[0, new]
            self.rank_inferences += 1
        self.committed = active
        order = self.ids.copy()
        for group in conflict_groups(first, second, core.num_agents):
            members = np.asarray(group)
            order[members] = members[np.lexsort((members, -self.scores[members]))]
        self.rank_changes += int(not np.array_equal(self.order, order))
        self.order = order


class TrackedMetrics(BaseMetrics):
    def summary(self):
        result = super().summary()
        result["rank_inferences"] = getattr(active_brake, "rank_inferences", 0)
        result["rank_changes"] = getattr(active_brake, "rank_changes", 0)
        return result


def initialize(config, flight, coordinator):
    pair.initialize(config, flight, coordinator, None, None)
    if pair._PAIR_SETTINGS.pairs != 64 or pair._PAIR_SETTINGS.refresh != .1:
        raise ValueError("Expected the original 64-pair, 0.1 s coordinator")
    evaluation.LayoutMetrics = TrackedMetrics


def run_case(layout, mode, output):
    def brake_factory(core):
        global active_brake
        active_brake = (PairRankedBrake(core) if mode == "pair_ranked_brake"
                        else priority_brake.PriorityBrake(core, CONFIG))
        return active_brake

    return evaluation.episode(AGENTS, layout, mode, output, brake_factory=brake_factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "coordinator", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    with (args.reference / "episodes.csv").open(newline="") as stream:
        ref = {r["layout_id"]: r for r in csv.DictReader(stream) if r["mode"] == "h065"}
    previous = json.loads((args.reference / "manifest.json").read_text())
    weight_inputs = [args.config, args.flight, args.coordinator, args.coordinator.with_name("evaluate_priority.py")]
    if any(pair.file_hash(p) != previous["files"][p.name] for p in weight_inputs):
        raise ValueError("The archived benchmark used different weights or evaluator")
    if len(layouts) != 16 or {layout["id"] for layout in layouts} != ref.keys():
        raise ValueError("Expected the original 16 matched scenes")
    inputs = [*weight_inputs, args.reference / "episodes.csv", args.reference / "manifest.json",
              Path(__file__), Path(pair.__file__), Path(pair_priority.__file__), Path(priority_brake.__file__)]
    manifest = dict(
        agents=AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION, modes=MODES,
        layouts=[layout["id"] for layout in layouts], brake=vars(CONFIG), pair_limit=64,
        rank_refresh_s=.1, coordinator_steps=100_000_000,
        ranking="static ID slots reordered within locked conflict components; rank scores fixed until release",
        files={p.name: pair.file_hash(p) for p in inputs}
        | {"evaluate_priority_with_brake_hook.py": pair.file_hash(Path(evaluation.__file__))},
    )
    manifest["modes"] = list(MODES)
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output contains a different experiment")
    evaluation.record_json(path, manifest)

    ordered = sorted(layouts, key=lambda layout: -float(ref[layout["id"]]["wall_seconds"]))
    tasks = list(product(ordered, MODES))
    print(f"Starting {len(tasks)} episodes with {args.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.flight, args.coordinator)) as pool:
        futures = [pool.submit(run_case, layout, mode, args.output) for layout, mode in tasks]
        for future in as_completed(futures):
            row = future.result()
            old = ref[row["layout_id"]]
            if row["layout_sha256"] != old["layout_sha256"] or row["seed"] != int(old["seed"]):
                raise ValueError(f"Layout differs: {row['layout_id']}")
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['layout_id']} {row['mode']} "
                  f"safe={row['success_rate']:.3f} changes={row['rank_changes']}", flush=True)
    evaluation.AGENT_COUNTS = (AGENTS,)
    evaluation.summarize(rows, args.output, layouts, MODES)
    print(f"Completed {len(rows)} matched episodes", flush=True)


if __name__ == "__main__":
    main()
