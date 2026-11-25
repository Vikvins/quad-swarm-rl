"""Compare the original pair-ranked brake with a trajectory-based yield choice."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path

from numba import njit
import numpy as np
from scipy.spatial import cKDTree

import evaluate_coordinator_next as original


ranking = original.ranking
evaluation = original.evaluation
pair = original.pair
predicts_conflict = original.brake.predicts_conflict
CONTROL = original.CONTROL
LOOKAHEAD = "pair_conflict_lookahead"
MODES = (CONTROL, LOOKAHEAD)


@njit(cache=True)
def trajectory_cost(paths, accepted, drone, hard_squared, buffer_squared, wall_hit):
    """Count predicted physical contacts first and near misses second."""
    physical = int(wall_hit)
    buffer_conflicts = 0
    for other in range(len(accepted)):
        if other == drone:
            continue
        if predicts_conflict(paths, accepted, drone, other, hard_squared):
            physical += 1
        elif predicts_conflict(paths, accepted, drone, other, buffer_squared):
            buffer_conflicts += 1
    return physical * (len(accepted) + 1) + buffer_conflicts


@njit(cache=True)
def choose_brakes(nominal, stopped, order, hard_squared, buffer_squared, wall_nominal, wall_stopped):
    """Compare keeping both trajectories, stopping the junior, and stopping the senior."""
    accepted = nominal.copy()
    braking = np.zeros(len(order), dtype=np.bool_)
    conflicts = 0
    reversals = 0
    unchanged = 0
    for rank in range(1, len(order)):
        junior = order[rank]
        for senior_rank in range(rank):
            senior = order[senior_rank]
            if not predicts_conflict(accepted, accepted, junior, senior, buffer_squared):
                continue
            conflicts += 1
            chosen = -1
            improvement = 0
            # The existing priority breaks ties between equally helpful changes.
            for drone in (junior, senior):
                if braking[drone]:
                    continue
                current = trajectory_cost(nominal, accepted, drone, hard_squared, buffer_squared, wall_nominal[drone])
                candidate = trajectory_cost(stopped, accepted, drone, hard_squared, buffer_squared, wall_stopped[drone])
                if current - candidate > improvement:
                    improvement = current - candidate
                    chosen = drone
            if chosen < 0:
                unchanged += 1
                continue
            accepted[chosen] = stopped[chosen]
            braking[chosen] = True
            reversals += int(chosen == senior)
    return braking, conflicts, reversals, unchanged


class LookaheadBrake(original.IntentPairBrake):
    def __init__(self, core):
        super().__init__(core)
        self.obstacles = None
        self.room = np.asarray(core.envs[0].room_box)
        self.hard_squared = core.collision_threshold ** 2
        self.conflicts = self.reversals = self.unchanged = 0

    def wall_hits(self, paths):
        plan = self.core.layout_plan
        clearance = np.minimum(paths - self.room[0], self.room[1] - paths).min(axis=-1)
        if len(plan.obstacles):
            if self.obstacles is None:
                self.obstacles = cKDTree(np.asarray(plan.obstacles)[:, :2])
            distance, _ = self.obstacles.query(paths[:, :, :2])
            clearance = np.minimum(clearance, distance - plan.spec.obstacle_size / 2)
        threshold = np.minimum(plan.spec.agent_radius, clearance[:, 0] - 1e-8)
        return clearance[:, 1:].min(axis=1) < threshold

    def decide(self):
        pos, vel = self.core.pos, self.core.vel
        goals = np.asarray([drone.goal for drone in self.core.envs])
        delta = goals - pos
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        intent = delta / np.maximum(distance, 1e-6)
        intent *= np.minimum(distance / original.HORIZON, .8)
        intent *= np.maximum(1.0 - speed / .8, 0.0)
        nominal = pos[:, None, :] + vel[:, None, :] * self.nominal_times + intent[:, None, :] * self.intent_times
        stopped = pos[:, None, :] + vel[:, None, :] * self.brake_times
        braking, conflicts, reversals, unchanged = choose_brakes(
            nominal, stopped, self.order_with_promotion(), self.hard_squared, self.threshold_squared,
            self.wall_hits(nominal), self.wall_hits(stopped),
        )
        self.conflicts += conflicts
        self.reversals += reversals
        self.unchanged += unchanged
        started = braking & (self.waited == 0)
        self.anchors[started] = pos[started]
        self.waited = np.where(braking, self.waited + 1, 0)
        self.brake_ticks += int(braking.sum())
        return braking


class ContactMetrics(ranking.TrackedMetrics):
    def update(self, positions, room_contacts):
        super().update(positions, room_contacts)
        if self.ticks <= round(1.5 / self.dt):
            contacts = self.neighbor_contacts | self.obstacle_contacts | self.room_contacts
            self.early_contacts = int(np.count_nonzero(contacts))

    def summary(self):
        result = super().summary()
        result["early_contacts"] = self.early_contacts
        for name in ("conflicts", "reversals", "unchanged"):
            result[name] = getattr(ranking.active_brake, name, 0)
        return result


def initialize(config, flight, coordinator):
    evaluation.initialize(config, flight, None)
    original.set_checkpoint(coordinator, 100_000_000)
    evaluation.LayoutMetrics = ContactMetrics


def run_scene(layout, mode, output):
    def controller(core):
        brake = LookaheadBrake if mode == LOOKAHEAD else original.IntentPairBrake
        ranking.active_brake = brake(core)
        return ranking.active_brake

    return evaluation.episode(original.AGENTS, layout, mode, output, brake_factory=controller)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "coordinator", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    reference_csv = args.reference / "episodes.csv"
    reference_manifest = args.reference / "manifest.json"
    with reference_csv.open(newline="") as stream:
        reference = {r["layout_id"]: r for r in csv.DictReader(stream)
                     if r["mode"] == "h065" and r["kind"] == "corridor"}
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"] and not layout["has_dead_end"]]
    if len(layouts) != 8 or {layout["id"] for layout in layouts} != reference.keys():
        raise ValueError("Expected the same eight corridor geometries")

    weights = {
        "baseline.json": pair.file_hash(args.config),
        "baseline.pth": pair.file_hash(args.flight),
        "coordinator_100m.pt": pair.file_hash(args.coordinator),
    }
    expected = json.loads(reference_manifest.read_text())["files"]
    if any(expected[name] != digest for name, digest in weights.items()):
        raise ValueError("Reference uses different flight or coordinator weights")
    inputs = Path(__file__), Path(original.__file__), Path(ranking.__file__), Path(original.brake.__file__)
    manifest = dict(agents=original.AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION,
                    layouts=sorted(reference), modes=list(MODES), brake=vars(ranking.CONFIG),
                    files=weights | {p.name: pair.file_hash(p) for p in inputs},
                    reference_sha256=pair.file_hash(reference_csv))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output directory belongs to different inputs")
    evaluation.record_json(manifest_path, manifest)

    tasks = list(product(sorted(layouts, key=lambda layout: -float(reference[layout["id"]]["wall_seconds"])), MODES))
    rows = []
    print(f"Starting {len(tasks)} matched corridor episodes with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.flight, args.coordinator)) as pool:
        pending = [pool.submit(run_scene, layout, mode, args.output) for layout, mode in tasks]
        for future in as_completed(pending):
            row = future.result()
            old = reference[row["layout_id"]]
            if row["layout_sha256"] != old["layout_sha256"] or row["seed"] != int(old["seed"]):
                raise ValueError(f"Starts or geometry changed for {row['layout_id']}")
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['layout_id']} {row['mode']} "
                  f"safe={row['success_rate']:.3f} reversals={row['reversals']}", flush=True)

    rows.sort(key=lambda row: (row["mode"], row["layout_id"]))
    evaluation.write_csv(args.output / "episodes.csv", rows)
    summary = []
    for mode in MODES:
        group = [row for row in rows if row["mode"] == mode]
        summary.append(dict(
            mode=mode, episodes=len(group), attempts=len(group) * original.AGENTS,
            safe_arrivals=sum(round(row["success_rate"] * original.AGENTS) for row in group),
            contacts=sum(round(row["collision_rate"] * original.AGENTS) for row in group),
            early_contacts=sum(row["early_contacts"] for row in group),
            timeouts=sum(round(row["timeout_rate"] * original.AGENTS) for row in group),
            hold_fraction=sum(row["hold_fraction"] for row in group) / len(group),
        ))
    evaluation.write_csv(args.output / "comparison.csv", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
