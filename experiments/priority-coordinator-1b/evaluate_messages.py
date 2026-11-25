"""Vary only the two received messages in the existing conflict-gated controller."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import evaluate_gated_priority as gated


evaluation = gated.evaluation
diagnose = gated.diagnose
MODES = ("messages_real", "messages_zero", "messages_shuffled")
AGENTS = 100
EARLY_SECONDS = 1.5
RELEASE_SPEED = .2
AWAY_FROM_GOAL = .5


class MessageAblation(gated.ConflictGatedPeerBrake):
    def __init__(self, core, mode, layout_id):
        super().__init__(core, gated.BrakeConfig())
        self.mode = mode
        seed = int.from_bytes(hashlib.sha256(f"{evaluation.SEED}/{layout_id}".encode()).digest()[:8], "big")
        order = np.random.default_rng(seed).permutation(core.num_agents)
        self.donor = np.empty_like(order)
        self.donor[order] = np.roll(order, -1)
        self.first_tick = True
        self.counts = dict(agent_ticks=0, early_agent_ticks=0, wait_requests=0, rejected_wait_requests=0,
                           zero_flips=0, shuffle_flips=0, zero_early_flips=0, shuffle_early_flips=0,
                           stationary_ticks_after_release=0, releases=0)
        self.previous_braking = np.zeros(core.num_agents, dtype=bool)
        self.stationary_after_release = self.previous_braking.copy()
        self.goals = None

    def prepare(self, obs):
        hidden, bids = self.score(obs)
        neighbors = torch.as_tensor(self.core.priority_neighbors, dtype=torch.long)
        received = bids[neighbors]
        flag = torch.ones_like(bids[:, None])

        def request(neighbor_bids):
            channels = torch.cat((bids[:, None], neighbor_bids), dim=-1).tanh() * self.policy.message_gain
            logits = self.policy.decide(torch.cat((hidden, channels, flag), dim=-1))
            return logits[:, 0].numpy() > 0

        if self.first_tick:
            exact_original = self.policy.wait_distribution(hidden, bids, neighbors, bids.new_ones((1,)))
            if not np.array_equal(request(received), exact_original.logits.numpy() > 0):
                raise RuntimeError("Original decision differs on real messages")
            self.goals = np.asarray([drone.goal for drone in self.core.envs])
            self.first_tick = False

        early = (self.tick - 1) * self.core.control_dt < EARLY_SECONDS
        self.counts["agent_ticks"] += self.core.num_agents
        self.counts["early_agent_ticks"] += self.core.num_agents * early
        if self.mode == "messages_zero":
            self.requests = request(torch.zeros_like(received))
        elif self.mode == "messages_shuffled":
            self.requests = request(received[self.donor])
        else:
            self.requests = request(received)
            zero_flips = int(np.count_nonzero(self.requests != request(torch.zeros_like(received))))
            shuffle_flips = int(np.count_nonzero(self.requests != request(received[self.donor])))
            self.counts["zero_flips"] += zero_flips
            self.counts["shuffle_flips"] += shuffle_flips
            self.counts["zero_early_flips"] += zero_flips * early
            self.counts["shuffle_early_flips"] += shuffle_flips * early
        self.counts["wait_requests"] += int(np.count_nonzero(self.requests))

    def decide(self):
        braking = super().decide()
        self.counts["rejected_wait_requests"] += int(np.count_nonzero(self.requests & ~braking))
        released = self.previous_braking & ~braking
        away = np.linalg.norm(self.goals - self.core.pos, axis=1) > AWAY_FROM_GOAL
        stationary = np.linalg.norm(self.core.vel, axis=1) < RELEASE_SPEED
        self.stationary_after_release = (self.stationary_after_release | released) & ~braking & away & stationary
        self.counts["releases"] += int(np.count_nonzero(released & away))
        self.counts["stationary_ticks_after_release"] += int(np.count_nonzero(self.stationary_after_release))
        self.previous_braking = braking
        return braking


class AblationMetrics(diagnose.TracedMetrics):
    def summary(self):
        row = super().summary()
        row.update(diagnose.trace["controller"].counts)
        t = np.asarray(row["first_contact_ticks"]["neighbor"], dtype=float) * self.dt
        row["first_neighbor_contact_by_1p5s"] = int(np.count_nonzero((0 <= t) & (t <= EARLY_SECONDS + 1e-9)))
        row["first_neighbor_contact_after_3s"] = int(np.count_nonzero(t > 3. + 1e-9))
        return row


def initialize(config, flight, peer):
    diagnose.init(config, flight, peer)
    evaluation.LayoutMetrics = AblationMetrics


def run_case(layout, mode, output):
    diagnose.trace = dict(signal=np.zeros(11), signal_ticks=0)

    def build(core):
        controller = MessageAblation(core, mode, layout["id"])
        diagnose.trace["controller"] = controller
        return controller

    return evaluation.episode(AGENTS, layout, mode, output, brake_factory=build)


def verify_reference(row, reference):
    old = reference[row["layout_id"]]
    if row["layout_sha256"] != old["layout_sha256"] or row["seed"] != int(old["seed"]):
        raise ValueError(f"Layout differs from original validation: {row['layout_id']}")
    if row["mode"] == "messages_real":
        for metric in ("success_rate", "collision_rate", "timeout_rate", "hold_fraction"):
            if abs(row[metric] - float(old[metric])) > 1e-12:
                raise ValueError(f"Real-message reference differs: {row['layout_id']} / {metric}")


def summarize(rows, output, layouts):
    discard = ("first_contact_ticks", "first_hold_ticks", "signal", "signal_ticks",
               "nearest_start", "start_span_xyz", "start_clearance", "collision_threshold")
    evaluation.AGENT_COUNTS = (AGENTS,)
    evaluation.summarize([{k: v for k, v in row.items() if k not in discard} for row in rows],
                         output, layouts, MODES)
    summary = []
    for kind in ("corridor", "dead_end"):
        for mode in MODES:
            group = [row for row in rows if (row["kind"], row["mode"]) == (kind, mode)]
            dt = group[0]["dt"]
            ticks = sum(row["agent_ticks"] for row in group)
            releases = sum(row["releases"] for row in group)
            summary.append(dict(
                kind=kind, mode=mode, episodes=len(group), drones=len(group) * AGENTS,
                safe_arrivals=sum(round(AGENTS * row["success_rate"]) for row in group),
                collisions=sum(round(AGENTS * row["collision_rate"]) for row in group),
                timeouts=sum(round(AGENTS * row["timeout_rate"]) for row in group),
                first_contacts_by_1p5s=sum(row["first_neighbor_contact_by_1p5s"] for row in group),
                first_contacts_after_3s=sum(row["first_neighbor_contact_after_3s"] for row in group),
                hold_fraction=mean(row["hold_fraction"] for row in group),
                rejected_request_fraction=sum(row["rejected_wait_requests"] for row in group) / ticks,
                release_events=releases,
                mean_stationary_seconds_per_release=(
                    dt * sum(row["stationary_ticks_after_release"] for row in group) / releases if releases else None),
            ))
    evaluation.write_csv(output / "comparison.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "peer", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    with args.reference.open(newline="") as source:
        reference = {row["layout_id"]: row for row in csv.DictReader(source) if row["mode"] == "gated_peer"}
    reference_manifest = json.loads((args.reference.parent / "manifest.json").read_text())
    inputs = (args.config, args.flight, args.peer, Path(gated.__file__),
              Path(evaluation.__file__), Path(diagnose.__file__),
              Path(evaluation.__file__).with_name("priority_brake.py"))
    hashes = {path.name: gated.file_hash(path) for path in inputs}
    if (set(reference) != {layout["id"] for layout in layouts}
            or any(reference_manifest["files"].get(name) != sha for name, sha in hashes.items())):
        raise ValueError("Original inputs or scenes differ")
    manifest = dict(
        agents=AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION,
        layouts=[layout["id"] for layout in layouts], modes=list(MODES),
        shuffled="fixed per-layout derangement of recipient rows of received message pairs",
        rank_and_own_bid="original in all modes", channel_flag=1,
        release_speed=RELEASE_SPEED, release_min_goal_distance=AWAY_FROM_GOAL,
        files={**hashes, Path(__file__).name: gated.file_hash(Path(__file__)),
               args.reference.name: gated.file_hash(args.reference)},
    )
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to mix different experiments")
    evaluation.record_json(path, manifest)
    tasks = [(layout, mode) for layout in layouts for mode in MODES]
    rows = []
    print(f"Starting {len(tasks)} matched episodes with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.flight, args.peer)) as pool:
        for future in as_completed([pool.submit(run_case, *task, args.output) for task in tasks]):
            row = future.result()
            verify_reference(row, reference)
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['layout_id']} {row['mode']} "
                  f"success={row['success_rate']:.3f} wall={row['wall_seconds']:.1f}s", flush=True)
    summarize(rows, args.output, layouts)
    print(f"Completed {len(rows)} matched episodes", flush=True)


if __name__ == "__main__":
    main()
