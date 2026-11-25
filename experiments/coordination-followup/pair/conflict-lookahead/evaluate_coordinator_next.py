"""Check the checkpoint curve and transfer goal-intent prediction to external ranks."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_rank_order as ranking
import validate_pair_horizon as horizon


evaluation = horizon.evaluation
pair = horizon.capacity.pair
brake = ranking.priority_brake
AGENTS = 100
HORIZON = .65
CONTROL = "intent_pair_ranked"
STEPS = (125_088_000, 150_073_600, 175_059_200, 200_000_000)
horizon.HORIZONS = {"h065": HORIZON}


class IntentPairBrake(ranking.PairRankedBrake):
    def __init__(self, core):
        super().__init__(core)
        self.intent_times = self.nominal_times - self.brake_times
        self.no_latched_hold = np.zeros(core.num_agents, dtype=bool)

    def decide(self):
        pos, vel = self.core.pos, self.core.vel
        goals = np.asarray([drone.goal for drone in self.core.envs])
        delta = goals - pos
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        intent = delta / np.maximum(distance, 1e-6)
        intent *= np.minimum(distance / HORIZON, .8)
        intent *= np.maximum(1.0 - speed / .8, 0.0)
        nominal = pos[:, None, :] + vel[:, None, :] * self.nominal_times + intent[:, None, :] * self.intent_times
        stopped = pos[:, None, :] + vel[:, None, :] * self.brake_times
        order = self.order_with_promotion()
        braking = brake.resolve_braking(nominal, stopped, order, self.no_latched_hold, self.threshold_squared)
        started = braking & (self.waited == 0)
        self.anchors[started] = pos[started]
        self.waited = np.where(braking, self.waited + 1, 0)
        self.brake_ticks += int(braking.sum())
        return braking


def initialize(config, flight, original, later):
    global original_path, later_dir
    original_path, later_dir = original, later
    evaluation.initialize(config, flight, None)


def set_checkpoint(path, steps):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = checkpoint["arguments"]
    if (checkpoint["progress"]["steps"] != steps or args["mode"] != "pair" or args.get("train_flight")
            or args["agents"] != AGENTS or args["pairs"] != 64 or args["refresh"] != .1):
        raise ValueError(f"Unexpected pair checkpoint: {path}")
    model = pair.PairPriority()
    model.load_state_dict(checkpoint["policy"], strict=True)
    pair._PAIR_MODEL = model.eval()
    pair._PAIR_SETTINGS = SimpleNamespace(**{key: args[key] for key in ("pairs", "horizon", "refresh")})
    assert ranking.pair is pair


def run_case(layout, variant, output):
    if variant == CONTROL:
        set_checkpoint(original_path, 100_000_000)
        evaluation.LayoutMetrics = ranking.TrackedMetrics

        def controller(core):
            ranking.active_brake = IntentPairBrake(core)
            return ranking.active_brake

        return evaluation.episode(AGENTS, layout, variant, output / variant, brake_factory=controller)

    steps = int(variant)
    set_checkpoint(later_dir / f"checkpoint_{steps:012d}.pt", steps)
    evaluation.LayoutMetrics = horizon.HorizonMetrics
    return horizon.run_scene(layout, "h065", output / variant)


def summarize(output, cases, variants):
    rows = sorted(cases, key=lambda r: (r["variant"], r["kind"], r["layout_id"]))
    base_keys = ("variant", "agents", "kind", "layout_id", "seed", "layout_sha256", "success_rate",
                 "collision_rate", "timeout_rate", "hold_fraction", "swarm_success", "wall_seconds")
    evaluation.write_csv(output / "episodes.csv", [{k: r[k] for k in base_keys} for r in rows])
    summary = []
    for variant in variants:
        for kind in ("corridor", "dead_end"):
            selected = [r for r in rows if r["variant"] == variant and r["kind"] == kind]
            if selected:
                if len(selected) != 8:
                    raise ValueError(f"Expected eight scenes for {variant} / {kind}")
                summary.append(dict(variant=variant, kind=kind, episodes=len(selected),
                                    safe_arrivals=sum(round(AGENTS * r["success_rate"]) for r in selected),
                                    contacts=sum(round(AGENTS * r["collision_rate"]) for r in selected),
                                    timeouts=sum(round(AGENTS * r["timeout_rate"]) for r in selected)))
    evaluation.write_csv(output / "comparison.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "coordinator", "checkpoints", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    with (args.reference / "episodes.csv").open(newline="") as f:
        ref = {r["layout_id"]: r for r in csv.DictReader(f) if r["mode"] == "h065"}
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    if len(ref) != 16 or set(ref) != {layout["id"] for layout in layouts}:
        raise ValueError("The archived comparison uses other scenes")
    metadata = json.loads((args.reference / "manifest.json").read_text())
    weights = (args.config, args.flight, args.coordinator)
    if any(pair.file_hash(p) != metadata["files"][p.name] for p in weights):
        raise ValueError("The archived comparison uses different weights")
    inputs = (*weights, Path(__file__), Path(horizon.__file__), Path(ranking.__file__),
              Path(brake.__file__), Path(evaluation.__file__), args.reference / "episodes.csv",
              *(args.checkpoints / f"checkpoint_{step:012d}.pt" for step in STEPS))
    manifest = dict(agents=AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION, horizon=HORIZON,
                    layouts=list(ref), checkpoint_steps=STEPS, intent_maximum_speed=.8,
                    intent_latched_hold=False, files={path.name: pair.file_hash(path) for path in inputs})
    manifest["checkpoint_steps"] = list(STEPS)
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output contains another experiment")
    evaluation.record_json(path, manifest)
    layouts.sort(key=lambda item: -float(ref[item["id"]]["wall_seconds"]))
    variants = (CONTROL, *map(str, STEPS))
    tasks = [(layout, CONTROL) for layout in layouts]
    tasks += list(product([layout for layout in layouts if not layout["has_dead_end"]], variants[1:]))
    print(f"Starting {len(tasks)} matched episodes with {args.workers} workers", flush=True)
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(*weights, args.checkpoints)) as pool:
        pending = {pool.submit(run_case, layout, variant, args.output): variant for layout, variant in tasks}
        for future in as_completed(pending):
            row = future.result()
            old = ref[row["layout_id"]]
            if row["layout_sha256"] != old["layout_sha256"] or row["seed"] != int(old["seed"]):
                raise ValueError(f"Scene differs from the original: {row['layout_id']}")
            variant = pending[future]
            cases.append(dict(row, variant=variant))
            print(f"{len(cases)}/{len(tasks)} {variant} {row['layout_id']} safe={row['success_rate']:.2f}", flush=True)
    summarize(args.output, cases, variants)
    print(f"Completed {len(cases)} matched episodes", flush=True)


if __name__ == "__main__":
    main()
