"""Compare both external coordinators and the gated channel on new matched starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path

import evaluate_conflict_lookahead as experiment
import evaluate_gated_priority as gated


SEEDS = (50001, 50002)
CHANNEL = "gated_peer"
MODES = (*experiment.MODES, CHANNEL)
evaluation = experiment.evaluation
AGENTS = experiment.original.AGENTS


def initialize(config, flight, coordinator, peer):
    evaluation.initialize(config, flight, None, peer)
    experiment.original.set_checkpoint(coordinator, 100_000_000)
    evaluation.LayoutMetrics = experiment.ContactMetrics


def run_scene(layout, mode, seed, output):
    evaluation.SEED = seed
    output = output / str(seed)
    if mode != CHANNEL:
        return experiment.run_scene(layout, mode, output)

    def factory(core):
        experiment.ranking.active_brake = gated.ConflictGatedPeerBrake(core, gated.BrakeConfig())
        return experiment.ranking.active_brake

    return evaluation.episode(AGENTS, layout, mode, output, brake_factory=factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "coordinator", "peer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    paths = (args.config, args.flight, args.coordinator, args.peer)
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"] and not layout["has_dead_end"]]
    if len(layouts) != 8 or gated.evaluation is not evaluation:
        raise ValueError("Expected one shared evaluator and eight corridor geometries")

    output = args.output
    manifest = dict(
        agents=AGENTS, duration=evaluation.DURATION, seeds=list(SEEDS), modes=list(MODES),
        layouts=[layout["id"] for layout in layouts],
        files={p.name: experiment.pair.file_hash(p) for p in
               (*paths, Path(__file__), Path(experiment.__file__), Path(gated.__file__))},
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Results use other inputs")
    evaluation.record_json(manifest_path, manifest)
    tasks = list(product(layouts, MODES, SEEDS))
    rows = []
    hashes = {}
    print(f"Starting {len(tasks)} corridor episodes on seeds {SEEDS}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=paths) as pool:
        pending = [pool.submit(run_scene, layout, mode, seed, output) for layout, mode, seed in tasks]
        for future in as_completed(pending):
            row = future.result()
            key = (row["layout_id"], row["seed"])
            digest = row["layout_sha256"]
            if hashes.setdefault(key, digest) != digest:
                raise ValueError(f"Different starts between modes for {key}")
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['seed']} {row['layout_id']} {row['mode']} "
                  f"safe={row['success_rate']:.3f}", flush=True)

    rows.sort(key=lambda row: (row["seed"], row["layout_id"], row["mode"]))
    evaluation.write_csv(output / "episodes.csv", rows)
    summary = []
    for seed_scope in (*SEEDS, "all"):
        for mode in MODES:
            group = [row for row in rows if row["mode"] == mode
                     and (seed_scope == "all" or row["seed"] == seed_scope)]
            summary.append(dict(
                seed=seed_scope, mode=mode, episodes=len(group), attempts=len(group) * AGENTS,
                safe_arrivals=sum(round(row["success_rate"] * AGENTS) for row in group),
                contacts=sum(round(row["collision_rate"] * AGENTS) for row in group),
                early_contacts=sum(row["early_contacts"] for row in group),
                timeouts=sum(round(row["timeout_rate"] * AGENTS) for row in group),
                hold_fraction=sum(row["hold_fraction"] for row in group) / len(group),
            ))
    evaluation.write_csv(output / "comparison.csv", summary)


if __name__ == "__main__":
    main()
