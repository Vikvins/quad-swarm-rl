"""Evaluate pair coordination and flight-only checkpoints on a shared scene suite."""

import argparse
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_priority as evaluation
from pair_priority import PairController, PairPriority, drone_features
from swarm_rl.evaluate_layouts import load_checkpoint


NEW_MODES = ("pair_learned", "pair_fixed_id", "baseline_trained8_100m", "baseline_trained100_100m")
REFERENCE_MODE = "baseline_trained100_1b"
MATCHED_FLIGHTS = {
    "pair_with_trained8_100m": "baseline_trained8_100m",
    "pair_with_trained100_100m": "baseline_trained100_100m",
}
_PAIR_MODEL = None
_PAIR_SETTINGS = None
_BASELINE_PATHS: Mapping[str, Path] = {}


def initialize(config, flight, coordinator, baseline8, baseline100):
    global _PAIR_MODEL, _PAIR_SETTINGS, _BASELINE_PATHS
    evaluation.initialize(config, flight, None)
    checkpoint = torch.load(coordinator, map_location="cpu", weights_only=True)
    if checkpoint["arguments"]["mode"] != "pair" or checkpoint["progress"]["steps"] != 100_000_000:
        raise ValueError("Expected the completed pair coordinator checkpoint at 100 million agent steps")
    _PAIR_MODEL = PairPriority()
    _PAIR_MODEL.load_state_dict(checkpoint["policy"], strict=True)
    _PAIR_MODEL.eval()
    _PAIR_SETTINGS = SimpleNamespace(**{key: checkpoint["arguments"][key] for key in ("pairs", "horizon", "refresh")})
    _BASELINE_PATHS = dict(baseline_trained8_100m=baseline8, baseline_trained100_100m=baseline100)


@lru_cache(maxsize=2)
def baseline_state(path):
    return load_checkpoint(path)["model"]


class PairEvaluation:
    def __init__(self, core, learned):
        self.core = core
        self.learned = learned
        self.control = PairController(1, core.num_agents, core.control_dt, core.collision_threshold, _PAIR_SETTINGS)
        self.initial = np.ones(1, dtype=bool)
        self.fixed_scores = -np.arange(core.num_agents, dtype=np.float32)[None]
        if not learned:
            self.control.scores[:] = self.fixed_scores
        self.yield_ticks = 0

    def policy_inputs(self, model, observations, states):
        core = self.core
        pos = np.asarray(core.pos, dtype=np.float32)[None]
        vel = np.asarray(core.vel, dtype=np.float32)[None]
        goals = np.asarray([agent.goal for agent in core.envs], dtype=np.float32)[None]
        pairs, active, decide = self.control.observe(pos, vel, self.initial)
        self.initial.fill(False)
        scores = self.fixed_scores
        if self.learned and decide.any():
            distribution, _ = _PAIR_MODEL(torch.from_numpy(drone_features(pos, vel, goals)), torch.from_numpy(pairs))
            scores = distribution.mean.numpy()
        offsets = self.control.offsets(scores, decide, active, pos, vel, goals)
        self.yield_ticks += int(self.control.previous_active.sum())
        features = torch.as_tensor(np.asarray(observations), dtype=torch.float32).clone()
        features[:, :3] += torch.from_numpy(offsets[0])
        return model.normalize_obs({"obs": features})


def run_case(agents, layout, mode, destination):
    path = _BASELINE_PATHS.get(MATCHED_FLIGHTS.get(mode, mode))
    flight_state = baseline_state(path) if path else None
    if mode in _BASELINE_PATHS:
        return evaluation.episode(agents, layout, mode, destination, flight_state=flight_state)
    return evaluation.episode(
        agents, layout, mode, destination, flight_state=flight_state,
        controller_factory=lambda core: PairEvaluation(core, learned=(mode != "pair_fixed_id")),
    )


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def reference_rows(path, inputs, layouts, mode_names):
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (manifest["seed"] != evaluation.SEED or manifest["duration"] != evaluation.DURATION
            or manifest["layouts"] != [layout["id"] for layout in layouts]
            or any(manifest["files"].get(item.name) != file_hash(item) for item in inputs)):
        raise ValueError("Reference baseline must use the same scenes, config, and flight checkpoint")
    with path.open(newline="") as source:
        return [dict(row, mode=mode_names[row["mode"]]) for row in csv.DictReader(source) if row["mode"] in mode_names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "coordinator", "baseline8", "baseline100", "reference", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--flight", type=Path, help="Frozen flight checkpoint in the original comparison")
    parser.add_argument("--matched-100m", action="store_true",
                        help="Evaluate the coordinator on each 100M flight checkpoint against its own baseline")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if not args.matched_100m and not args.flight:
        parser.error("The original comparison requires --flight")
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"] is not None]
    if len(layouts) != 16:
        raise ValueError("Expected eight corridors and eight paired dead ends")
    if args.matched_100m:
        new_modes = tuple(MATCHED_FLIGHTS)
        reference_names = {mode: mode for mode in MATCHED_FLIGHTS.values()}
        reference_inputs = (args.config, args.baseline8, args.baseline100)
        flight = args.baseline100
    else:
        new_modes = NEW_MODES
        reference_names = {"baseline": REFERENCE_MODE}
        reference_inputs = (args.config, args.flight)
        flight = args.flight
    references = reference_rows(args.reference, reference_inputs, layouts, reference_names)
    modes = (*new_modes, *reference_names.values())
    reference_manifest = args.reference.parent / "manifest.json"
    inputs = (args.config, flight, args.coordinator, args.baseline8, args.baseline100,
              args.reference, reference_manifest, Path(__file__), Path(evaluation.__file__),
              Path(__file__).with_name("pair_priority.py"), Path(__file__).with_name("train_priority.py"))
    manifest = dict(
        agents=list(evaluation.AGENT_COUNTS), modes=list(modes), seed=evaluation.SEED,
        duration=evaluation.DURATION, layouts=[layout["id"] for layout in layouts], learned_priorities="mean",
        reference_modes=list(reference_names.values()), files={path.name: file_hash(path) for path in inputs},
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to merge results from different configurations")
    evaluation.record_json(manifest_path, manifest)

    tasks = [(n, layout, mode, args.output) for n in reversed(evaluation.AGENT_COUNTS)
             for layout in layouts for mode in new_modes]
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=initialize, initargs=(args.config, flight, args.coordinator, args.baseline8, args.baseline100),
    ) as pool:
        futures = [pool.submit(run_case, *task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} N={row['agents']} {row['layout_id']} {row['mode']} "
                  f"success={row['success_rate']:.3f} wall={row['wall_seconds']:.1f}s", flush=True)

    if not references or set(references[0]) != set(rows[0]):
        raise ValueError("Reference baseline row schema differs from the current evaluation")
    for row in references:
        for key, sample in rows[0].items():
            if row[key] == "":
                row[key] = None
            elif isinstance(sample, int):
                row[key] = int(row[key])
            elif sample is None or isinstance(sample, float):
                row[key] = float(row[key])
    evaluation.summarize([*rows, *references], args.output, layouts, modes)
    print(f"{len(rows)} new episodes and {len(references)} matched reference episodes.", flush=True)


if __name__ == "__main__":
    main()
