"""Paired sensor outages with fixed maps and independent noise repetitions."""

import argparse
from collections import defaultdict
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

import numpy as np
import torch

import evaluate_capacity
from evaluate_capacity import evaluate


SIZES = (200, 300)
SCENES = range(50000, 50016)
REPEATS = (1, 2, 3)
CONDITIONS = dict(evaluate.CONDITIONS)


class NoiseRepeat(evaluate.Evaluation):
    def __init__(self, config, agents, scene_seed, noise_seed):
        super().__init__(config, agents)
        rng = np.random.default_rng(scene_seed)
        rng.choice(agents, agents // 4, replace=False)
        self.fault_tick = round(rng.uniform(3, 8) * self.core.control_freq)
        self.pre_fault = hashlib.sha256()
        self.observed_ticks = 0
        original_reset = self.core.reset
        original_step = self.core.step

        def update(value):
            self.pre_fault.update(np.ascontiguousarray(value).tobytes())

        def reset(*args, **kwargs):
            obs = original_reset(*args, **kwargs)
            if not self.capture_terminal:
                for drone in self.core.envs:
                    drone.dynamics.thrust_noise.reset()
                random.seed(noise_seed)
                np.random.seed(noise_seed)
                evaluate.seed_numba(noise_seed)
                torch.manual_seed(noise_seed)
                update(obs)
                update(self.core.pos)
                update(self.core.vel)
            return obs

        def step(action):
            result = original_step(action)
            tick = self.core.envs[0].tick
            if 0 < tick <= self.fault_tick:
                update(action)
                update(self.core.pos)
                update(self.core.vel)
                self.observed_ticks += 1
            return result

        self.core.reset = reset
        self.core.step = step


def initialize(config, checkpoint, reference, output):
    global CONFIG, CHECKPOINT, REFERENCE, OUTPUT
    CONFIG, CHECKPOINT, REFERENCE, OUTPUT = config, checkpoint, reference, output
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluate.register_swarm_components()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def noise_seed_for(scene_seed, repeat):
    return int(np.random.SeedSequence([20261124, scene_seed, repeat]).generate_state(1)[0])


def destination(output, n, scene, repeat, mode):
    return output / "episodes" / f"n{n}-seed{scene}-repeat{repeat}-{mode}.json"


def trial(n, scene_seed, repeat, mode):
    noise_seed = noise_seed_for(scene_seed, repeat)
    with open(os.devnull, "w") as silent, redirect_stdout(silent):
        runner = NoiseRepeat(CONFIG, n, scene_seed, noise_seed)
        try:
            row = runner.episode(str(CHECKPOINT), scene_seed, mode, CONDITIONS[mode])
            if runner.observed_ticks != runner.fault_tick:
                raise RuntimeError(f"Incomplete pre-fault trace for {n}/{scene_seed}/{repeat}/{mode}")
        finally:
            runner.env.close()
    if row["scene_id"] != REFERENCE[n, scene_seed]:
        raise RuntimeError(f"Scene changed for {n}/{scene_seed}: {row['scene_id']}")
    row.update(repetition=repeat, noise_seed=noise_seed, pre_fault_sha256=runner.pre_fault.hexdigest(),
               planned_fault_start_tick=runner.fault_tick)
    atomic_json(destination(OUTPUT, n, scene_seed, repeat, mode), row)
    return row


def match_group(group):
    if len(group) != len(CONDITIONS) or set(group) != set(CONDITIONS):
        raise ValueError("Missing or duplicate conditions in one scene/noise group")
    for key in ("pre_fault_sha256", "scene_id", "noise_seed", "planned_fault_start_tick"):
        if len({row[key] for row in group.values()}) != 1:
            raise RuntimeError(f"Paired runs differ in {key}: {[(r['agents'], r['seed'], r['repetition'], r['condition']) for r in group.values()]}")


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(output, paired):
    rows = [group[mode] for _, group in sorted(paired.items()) for mode in CONDITIONS]
    write_csv(output / "episodes.csv", rows)
    stats = []
    for n in SIZES:
        for mode in CONDITIONS:
            run = [r for r in rows if r["agents"] == n and r["condition"] == mode]
            stats.append(dict(agents=n, condition=mode, scenes=len(SCENES), repetitions=len(REPEATS),
                              episodes=len(run), agent_success=mean(r["agent_success"] for r in run),
                              selected_agent_success=mean(r["selected_agent_success"] for r in run),
                              healthy_agent_success=mean((4*r["agent_success"] - r["selected_agent_success"]) / 3
                                                        for r in run),
                              swarm_success=mean(r["swarm_success"] for r in run)))
    write_csv(output / "summary.csv", stats)

    tests = []
    for n in SIZES:
        for mode, duration in CONDITIONS.items():
            if duration == 0:
                continue
            for group in ("all", "selected", "healthy"):
                sizes = {"all": n, "selected": n // 4, "healthy": 3 * n // 4}
                deltas = []
                for scene in SCENES:
                    difference = 0
                    for repeat in REPEATS:
                        pair = paired[n, scene, repeat]
                        def successes(row):
                            total = round(n * row["agent_success"])
                            selected = round(n // 4 * row["selected_agent_success"])
                            return {"all": total, "selected": selected, "healthy": total-selected}[group]
                        difference += successes(pair[mode]) - successes(pair["clean"])
                    deltas.append(difference)
                null = [0]
                for delta in deltas:
                    null = [item + sign * delta for item in null for sign in (-1, 1)]
                observed = sum(deltas)
                p = sum(abs(item) >= abs(observed) for item in null) / len(null)
                tests.append(dict(agents=n, condition=mode, outcome=group, scenes=len(SCENES),
                                  delta_pp=100 * observed / (sizes[group] * len(SCENES) * len(REPEATS)),
                                  p_value=p, holm_p_value=0.))
    adjusted = 0.
    for i, row in enumerate(sorted(tests, key=lambda r: r["p_value"])):
        adjusted = max(adjusted, min(1., (len(tests) - i) * row["p_value"]))
        row["holm_p_value"] = adjusted
    write_csv(output / "paired-tests.csv", tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    refs = {}
    for n in SIZES:
        source = args.reference / f"n{n}-episodes.csv"
        with source.open() as f:
            refs.update({(n, int(row["seed"])): row["scene_id"] for row in csv.DictReader(f)
                         if row["condition"] == "clean"})
    if set(refs) != {(n, seed) for n in SIZES for seed in SCENES}:
        raise ValueError("The reference files must contain the 16 original scenes per size")

    (args.output / "episodes").mkdir(parents=True, exist_ok=True)
    metadata = dict(agents=list(SIZES), scene_seeds=list(SCENES), repeats=list(REPEATS),
                    conditions=CONDITIONS, noise_seed_material=20261124,
                    files={f.name:file_hash(f)
                           for f in (args.config,args.checkpoint,Path(__file__),
                                     Path(evaluate.__file__),Path(evaluate_capacity.__file__))},
                    reference_scenes={str(n): {str(seed):refs[n,seed] for seed in SCENES} for n in SIZES})
    manifest = args.output / "manifest.json"
    if manifest.is_file() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Output directory contains a different experiment")
    atomic_json(manifest, metadata)

    tasks = [(n, scene, repeat, mode) for repeat in REPEATS for scene in SCENES
             for mode in CONDITIONS for n in SIZES]
    groups, pending = defaultdict(dict), []
    counts = defaultdict(int)
    def record(row):
        n, scene, repeat, mode = (row[k] for k in ("agents", "seed", "repetition", "condition"))
        group = groups[n, scene, repeat]
        group[mode] = row
        if len(group) == len(CONDITIONS):
            match_group(group)
        counts[n] += 1
        print(f"n{n} {counts[n]}/144 | scene {scene} | repeat {repeat} | {mode} | {row['agent_success']:.4f}",
              flush=True)

    for task in tasks:
        path = destination(args.output, *task)
        if path.is_file():
            row = json.loads(path.read_text())
            if tuple(row[k] for k in ("agents","seed","repetition","condition")) != task:
                raise ValueError(f"Cached episode key differs from its path: {path}")
            record(row)
        else:
            pending.append(task)
    print(f"Starting {len(pending)} remaining episodes with {args.workers} workers", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize, initargs=(args.config,args.checkpoint,refs,args.output)) as pool:
            futures = [pool.submit(trial, *task) for task in pending]
            for future in as_completed(futures):
                record(future.result())
    if len(groups) != 96 or any(counts[n] != 144 for n in SIZES):
        raise RuntimeError(f"Incomplete run: {dict(counts)}")
    summarize(args.output, groups)
    print("Completed 288 episodes; all 96 three-condition groups verified.", flush=True)


if __name__ == "__main__":
    main()

