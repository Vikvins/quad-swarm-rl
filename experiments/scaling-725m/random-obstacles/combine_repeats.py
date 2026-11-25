"""Combine completed repeat runs and correct the tests across all swarm sizes."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from evaluate_repeats import (
    CONDITIONS,
    REPEATS,
    SCENES,
    atomic_json,
    match_group,
    noise_seed_for,
    summarize,
    write_csv,
)
from summarize_variance import summarize as summarize_variance


def read_batches(sources):
    groups = defaultdict(dict)
    runs = []
    reference_scenes = {}
    common_files = None
    for source in sources:
        manifest = json.loads((source / "manifest.json").read_text())
        if (manifest["scene_seeds"] != list(SCENES) or manifest["repeats"] != list(REPEATS)
                or manifest["conditions"] != CONDITIONS or manifest["noise_seed_material"] != 20261124):
            raise ValueError(f"Different validation protocol in {source}")
        files = {name: sha for name, sha in manifest["files"].items() if name != "evaluate_repeats.py"}
        if common_files is None:
            common_files = files
        elif files != common_files:
            raise ValueError(f"Different checkpoint, config or simulator helpers in {source}")
        scene_refs = manifest["reference_scenes"]
        agents = manifest["agents"]
        if set(scene_refs) != set(map(str, agents)) or set(scene_refs) & set(reference_scenes):
            raise ValueError(f"Duplicate or missing swarm sizes in {source}")
        reference_scenes.update(scene_refs)
        runs.append(manifest)

        expected = {(n, seed, repeat, mode) for n in agents for seed in SCENES
                    for repeat in REPEATS for mode in CONDITIONS}
        with (source / "episodes.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                for key in ("agents", "seed", "repetition", "noise_seed", "swarm_success", "planned_fault_start_tick"):
                    row[key] = int(row[key])
                for key in ("agent_success", "selected_agent_success"):
                    row[key] = float(row[key])
                n, seed, repeat, mode = (row[k] for k in ("agents", "seed", "repetition", "condition"))
                expected.remove((n, seed, repeat, mode))
                if (row["scene_id"] != scene_refs[str(n)][str(seed)]
                        or row["noise_seed"] != noise_seed_for(seed, repeat)):
                    raise ValueError(f"Scene or noise seed differs in {source} / {n} / {seed} / {repeat}")
                groups[n, seed, repeat][mode] = row
        if expected:
            raise ValueError(f"Missing {len(expected)} episodes in {source}")

    for group in groups.values():
        match_group(group)
    metadata = dict(agents=sorted(map(int, reference_scenes)), scene_seeds=list(SCENES),
                    repeats=list(REPEATS), conditions=CONDITIONS, noise_seed_material=20261124,
                    reference_scenes=reference_scenes, runs=runs)
    return groups, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups, metadata = read_batches(args.sources)
    args.output.mkdir(parents=True, exist_ok=True)
    summarize(args.output, groups)
    episodes = (row for group in groups.values() for row in group.values())
    write_csv(args.output / "variance.csv", list(summarize_variance(episodes, metadata)))
    atomic_json(args.output / "manifest.json", metadata)
    print(f"Combined {len(groups)} matched three-condition groups across {len(metadata['agents'])} swarm sizes.")


if __name__ == "__main__":
    main()
