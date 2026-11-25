"""Compare the completed pair coordinator on the large structured layouts."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean

import evaluate_pair_checkpoint as pair
import evaluate_priority as evaluation
from gym_art.quadrotor_multi.obstacles import suite_layouts


SIZES = (200, 300)
SEEDS = (50000,)
MODES = ("baseline", "static_brake", "pair_fixed_id", "pair_learned")
METRICS = ("success_rate", "reached_goal_rate", "collision_rate", "neighbor_collision_rate",
           "obstacle_collision_rate", "room_collision_rate", "timeout_rate", "hold_fraction")


def run_case(seed, layout, mode, agents, output):
    evaluation.SEED = seed
    destination = output / f"seed-{seed}"
    if mode in ("pair_fixed_id", "pair_learned"):
        return pair.run_case(agents, layout, mode, destination)
    return evaluation.episode(agents, layout, mode, destination)


def summarize(rows, output, layouts):
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        evaluation.summarize(seed_rows, output / f"seed-{seed}", layouts, MODES)
    summary = []
    for agents, kind, mode in product(SIZES, ("corridor", "dead_end"), MODES):
        selected = [row for row in rows if (row["agents"], row["kind"], row["mode"]) == (agents, kind, mode)]
        summary.append(dict(agents=agents, kind=kind, mode=mode, episodes=len(selected),
                            **{key: mean(row[key] for row in selected) for key in METRICS},
                            full_swarm_successes=sum(row["swarm_success"] for row in selected)))
    evaluation.write_csv(output / "summary.csv", summary)
    evaluation.write_csv(output / "episodes.csv", sorted(rows, key=lambda row: (
        row["agents"], row["kind"], row["layout_id"], row["seed"], row["mode"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "flight", "coordinator", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    evaluation.AGENT_COUNTS = SIZES
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"] is not None]
    if Counter(layout["has_dead_end"] for layout in layouts) != {False: 8, True: 8}:
        raise ValueError("Expected eight corridors and eight paired dead ends")
    provenance = (args.config, args.flight, args.coordinator, Path(__file__),
                  Path(pair.__file__), Path(evaluation.__file__), Path(suite_layouts.__file__),
                  Path(__file__).with_name("pair_priority.py"), Path(__file__).with_name("priority_brake.py"))
    manifest = dict(
        agents=list(SIZES), seeds=list(SEEDS), modes=list(MODES), duration=evaluation.DURATION,
        layouts=[layout["id"] for layout in layouts],
        simulator_revision="7f4596ee06d9d55dcb98ed985ab4a4fcc025fc20",
        spawn_capacity_source="scaling-725m-20261123",
        learned_priorities="mean", files={p.name: pair.file_hash(p) for p in provenance},
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing results use another configuration")
    evaluation.record_json(manifest_path, manifest)
    tasks = list(product(SEEDS, layouts, MODES, SIZES))
    per_size = len(tasks) // len(SIZES)
    counts = Counter()
    hashes = {}
    rows = []
    print(f"Starting {len(tasks)} episodes with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=pair.initialize,
                             initargs=(args.config, args.flight, args.coordinator, None, None)) as pool:
        futures = [pool.submit(run_case, *task, args.output) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            key = (row["agents"], row["layout_id"], row["seed"])
            if hashes.setdefault(key, row["layout_sha256"]) != row["layout_sha256"]:
                raise RuntimeError(f"Inconsistent start or layout across modes for {key}")
            rows.append(row)
            counts[row["agents"]] += 1
            print(f"{len(rows)}/{len(tasks)} | N={key[0]} {counts[key[0]]}/{per_size} | {key[1]} seed={key[2]} "
                  f"{row['mode']} success={row['success_rate']:.4f} wall={row['wall_seconds']:.1f}s", flush=True)
    summarize(rows, args.output, layouts)
    print(f"Completed {len(rows)} episodes", flush=True)


if __name__ == "__main__":
    main()
