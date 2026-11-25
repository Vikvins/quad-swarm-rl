"""A matched validation of the fixed 100M coordinator at different pair limits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import validate_pair_200m as common
from swarm_rl.dynamic_priority import closest_distance_squared


pair = common.pair
evaluation = common.evaluation
BaseMetrics = evaluation.LayoutMetrics
PAIR_LIMITS = (64, 128, 256)
MODES = tuple(f"k{limit}" for limit in PAIR_LIMITS)
AGENTS = 100
EARLY_SECONDS = 1.5


class CapacityMetrics(BaseMetrics):
    def summary(self):
        return dict(super().summary(), **counters)


def initialize(config, flight, coordinator):
    common.STEPS = 100_000_000
    common.initialize(config, flight, coordinator)
    if pair._PAIR_SETTINGS.pairs != PAIR_LIMITS[0]:
        raise ValueError("Baseline must use the training-time limit of 64 pairs")
    evaluation.LayoutMetrics = CapacityMetrics


def tracked_controller(core, limit):
    evaluator = pair.PairEvaluation(core, learned=True)
    settings = SimpleNamespace(**(vars(pair._PAIR_SETTINGS) | {"pairs": limit}))
    control = pair.PairController(1, core.num_agents, core.control_dt, core.collision_threshold, settings)
    evaluator.control = control
    original_observe = control.observe

    def observe(pos, vel, ended):
        measured_time = counters["pair_observation_ticks"] * core.control_dt
        pairs, active, decide = original_observe(pos, vel, ended)
        counters["pair_observation_ticks"] += 1
        counters["active_pairs_total"] += int(active.sum())
        if active.all():
            counters["full_candidate_ticks"] += 1
            difference = pos[:, control.first] - pos[:, control.second]
            relative_velocity = vel[:, control.first] - vel[:, control.second]
            distances = closest_distance_squared(difference, relative_velocity, control.horizon)
            omitted = max(0, int(np.count_nonzero(distances < control.safe_sq)) - limit)
            if omitted:
                counters["overflow_ticks"] += 1
                counters["early_overflow_ticks"] += int(measured_time < EARLY_SECONDS)
                counters["omitted_dangerous_pairs"] += omitted
                counters["max_omitted_dangerous_pairs"] = max(counters["max_omitted_dangerous_pairs"], omitted)
        return pairs, active, decide

    control.observe = observe
    return evaluator


def run_scene(layout, limit, output):
    global counters
    counters = dict(pair_limit=limit, pair_observation_ticks=0, active_pairs_total=0,
                    full_candidate_ticks=0, overflow_ticks=0, early_overflow_ticks=0,
                    omitted_dangerous_pairs=0, max_omitted_dangerous_pairs=0)
    return evaluation.episode(AGENTS, layout, f"k{limit}", output,
                              controller_factory=lambda core: tracked_controller(core, limit))


def verify(row, reference):
    expected = reference[row["layout_id"]]
    if int(expected["seed"]) != row["seed"] or expected["layout_sha256"] != row["layout_sha256"]:
        raise ValueError(f"Layout differs from reference: {row['layout_id']}")
    if row["pair_limit"] != PAIR_LIMITS[0]:
        return []
    return [
        dict(layout_id=row["layout_id"], metric=metric, reference=float(expected[metric]), current=row[metric])
        for metric in ("success_rate", "collision_rate", "timeout_rate", "hold_fraction")
        if abs(float(expected[metric]) - row[metric]) > 1e-12
    ]


def write_comparison(rows, output):
    data = []
    for kind in ("corridor", "dead_end"):
        for limit in PAIR_LIMITS:
            selected = [row for row in rows if (row["kind"], row["pair_limit"]) == (kind, limit)]
            total_ticks = sum(row["pair_observation_ticks"] for row in selected)
            data.append(dict(
                kind=kind, pair_limit=limit, episodes=len(selected), attempts=AGENTS * len(selected),
                safe_arrivals=sum(round(AGENTS * row["success_rate"]) for row in selected),
                collisions=sum(round(AGENTS * row["collision_rate"]) for row in selected),
                timeouts=sum(round(AGENTS * row["timeout_rate"]) for row in selected),
                overflow_ticks=sum(row["overflow_ticks"] for row in selected),
                early_overflow_ticks=sum(row["early_overflow_ticks"] for row in selected),
                pair_observation_ticks=total_ticks,
                maximum_omitted_pairs=max(row["max_omitted_dangerous_pairs"] for row in selected),
            ))
    evaluation.write_csv(output / "comparison.csv", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "coordinator", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    with args.reference.open(newline="") as source:
        reference = {row["layout_id"]: row for row in csv.DictReader(source)
                     if row["agents"] == str(AGENTS) and row["mode"] == "pair_learned"}
    if len(reference) != 16 or set(reference) != {layout["id"] for layout in layouts}:
        raise ValueError("The reference should contain eight corridors and eight dead ends")
    inputs = (args.config, args.flight, args.coordinator, args.reference, Path(__file__),
              Path(evaluation.__file__), Path(pair.__file__), Path(pair.__file__).with_name("pair_priority.py"))
    manifest = dict(agents=AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION, pair_limits=list(PAIR_LIMITS),
                    layout_ids=[layout["id"] for layout in layouts], early_seconds=EARLY_SECONDS,
                    files={path.name: pair.file_hash(path) for path in inputs})
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing outputs use different weights, scenes or settings")
    evaluation.record_json(path, manifest)
    tasks = [(layout, limit) for layout in layouts for limit in PAIR_LIMITS]
    rows = []
    differences = []
    print(f"Starting {len(tasks)} matched episodes with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.flight, args.coordinator)) as pool:
        futures = [pool.submit(run_scene, layout, limit, args.output) for layout, limit in tasks]
        for future in as_completed(futures):
            row = future.result()
            differences.extend(verify(row, reference))
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['layout_id']} K={row['pair_limit']} "
                  f"safe={row['success_rate']:.3f} overflow_ticks={row['overflow_ticks']}", flush=True)
    evaluation.AGENT_COUNTS = (AGENTS,)
    evaluation.summarize(rows, args.output, layouts, MODES)
    write_comparison(rows, args.output)
    evaluation.record_json(args.output / "reference-differences.json", differences)
    print(f"Completed {len(rows)} matched episodes; reference metric differences: {len(differences)}", flush=True)


if __name__ == "__main__":
    main()
