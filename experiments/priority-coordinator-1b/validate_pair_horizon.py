"""Vary the conflict prediction horizon for the frozen-flight pair coordinator."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean

import numpy as np

import validate_pair_capacity as capacity


evaluation = capacity.evaluation
AGENTS = capacity.AGENTS
LIMIT = 64
HORIZONS = {"h025": .25, "h050": .5, "h065": .65}


class HorizonMetrics(capacity.CapacityMetrics):
    def update(self, positions, room_contacts):
        super().update(positions, room_contacts)
        if self.ticks <= round(capacity.EARLY_SECONDS / self.dt):
            all_contacts = self.neighbor_contacts | self.obstacle_contacts | self.room_contacts
            capacity.counters["early_contacts"] = int(np.count_nonzero(all_contacts))
            capacity.counters["early_neighbor_contacts"] = int(np.count_nonzero(self.neighbor_contacts))


def initialize(config, flight, coordinator):
    capacity.initialize(config, flight, coordinator)
    evaluation.LayoutMetrics = HorizonMetrics


def run_scene(layout, mode, output):
    capacity.counters = dict(horizon=HORIZONS[mode], pair_limit=LIMIT, pair_observation_ticks=0,
                             active_pairs_total=0, full_candidate_ticks=0, overflow_ticks=0, early_overflow_ticks=0,
                             omitted_dangerous_pairs=0, max_omitted_dangerous_pairs=0,
                             early_contacts=0, early_neighbor_contacts=0)

    def controller(core):
        tracked = capacity.tracked_controller(core, LIMIT)
        tracked.control.horizon = HORIZONS[mode]
        return tracked

    return evaluation.episode(AGENTS, layout, mode, output, controller_factory=controller)


def summarize(rows, output, layouts):
    evaluation.AGENT_COUNTS = (AGENTS,)
    evaluation.summarize(rows, output, layouts, HORIZONS)
    summary = []
    for kind in ("corridor", "dead_end"):
        for mode, horizon in HORIZONS.items():
            group = [row for row in rows if (row["kind"], row["mode"]) == (kind, mode)]
            summary.append(dict(
                kind=kind, horizon_s=horizon, episodes=len(group), attempts=len(group) * AGENTS,
                safe_arrivals=sum(round(AGENTS * r["success_rate"]) for r in group),
                collisions=sum(round(AGENTS * r["collision_rate"]) for r in group),
                timeouts=sum(round(AGENTS * r["timeout_rate"]) for r in group),
                contacts_by_1p5s=sum(r["early_contacts"] for r in group),
                neighbor_contacts_by_1p5s=sum(r["early_neighbor_contacts"] for r in group),
                overflow_ticks=sum(r["overflow_ticks"] for r in group),
                hold_fraction=mean(r["hold_fraction"] for r in group),
            ))
    evaluation.write_csv(output / "comparison.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "coordinator", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    if len(layouts) != 16:
        raise ValueError("Expected eight corridors and eight dead ends")
    reference = {layout["id"]: json.loads((args.reference / "scenes" / "n100" / layout["id"] / "k64.json").read_text())
                 for layout in layouts}
    files = (args.config, args.flight, args.coordinator, Path(__file__), Path(capacity.__file__),
             Path(evaluation.__file__), args.reference / "manifest.json")
    manifest = dict(agents=AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION, pair_limit=LIMIT,
                    horizons=HORIZONS, layouts=[layout["id"] for layout in layouts],
                    files={path.name: capacity.pair.file_hash(path) for path in files})
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output contains a different run")
    evaluation.record_json(path, manifest)
    tasks = [(layout, mode) for layout in layouts for mode in HORIZONS]
    print(f"Starting {len(tasks)} matched episodes with {args.workers} workers", flush=True)
    rows = []
    diffs = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.flight, args.coordinator)) as pool:
        futures = [pool.submit(run_scene, layout, mode, args.output) for layout, mode in tasks]
        for future in as_completed(futures):
            row = future.result()
            old = reference[row["layout_id"]]
            if row["layout_sha256"] != old["layout_sha256"] or row["seed"] != old["seed"]:
                raise ValueError(f"Layout differs from reference: {row['layout_id']}")
            if row["mode"] == "h025":
                for metric in ("success_rate", "collision_rate", "timeout_rate", "hold_fraction"):
                    if abs(row[metric] - old[metric]) > 1e-12:
                        diffs.append(dict(layout_id=row["layout_id"], metric=metric,
                                          old=old[metric], new=row[metric]))
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['layout_id']} h={row['horizon']:.2f} "
                  f"safe={row['success_rate']:.3f} early_neighbor={row['early_neighbor_contacts']}", flush=True)
    summarize(rows, args.output, layouts)
    evaluation.record_json(args.output / "reference-differences.json", diffs)
    print(f"Completed {len(rows)} matched episodes; reference metric differences: {len(diffs)}", flush=True)


if __name__ == "__main__":
    main()
