"""Evaluate the 200M frozen-flight coordinator on the original 100-agent scenes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import torch

import evaluate_pair_checkpoint as pair


evaluation = pair.evaluation
AGENTS = 100
MODE = "pair_learned"
STEPS = 200_000_000


def initialize(config, flight, coordinator):
    evaluation.initialize(config, flight, None)
    checkpoint = torch.load(coordinator, map_location="cpu", weights_only=True)
    arguments = checkpoint["arguments"]
    if (checkpoint["progress"]["steps"] != STEPS or arguments["mode"] != "pair"
            or arguments.get("train_flight") or arguments["agents"] != AGENTS):
        raise ValueError("Expected the 200M coordinator trained with frozen flight")
    pair._PAIR_MODEL = pair.PairPriority()
    pair._PAIR_MODEL.load_state_dict(checkpoint["policy"], strict=True)
    pair._PAIR_MODEL.eval()
    pair._PAIR_SETTINGS = SimpleNamespace(**{key: arguments[key] for key in ("pairs", "horizon", "refresh")})


def evaluate_scene(layout, output):
    return evaluation.episode(AGENTS, layout, MODE, output,
                              controller_factory=lambda core: pair.PairEvaluation(core, learned=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "flight", "coordinator", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    with args.reference.open(newline="") as source:
        reference = {r["layout_id"]: r for r in csv.DictReader(source)
                     if r["agents"] == str(AGENTS) and r["mode"] == MODE}
    if len(reference) != 16 or set(reference) != {layout["id"] for layout in layouts}:
        raise ValueError("Expected the original 16 matched scenes")
    inputs = (args.config, args.flight, args.coordinator, args.reference, Path(__file__), Path(pair.__file__),
              Path(evaluation.__file__), Path(pair.__file__).with_name("pair_priority.py"))
    manifest = dict(
        agents=AGENTS, seed=evaluation.SEED, duration=evaluation.DURATION, coordinator_steps=STEPS,
        layouts=[layout["id"] for layout in layouts], files={path.name: pair.file_hash(path) for path in inputs},
    )
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing evaluation uses other weights or scenes")
    evaluation.record_json(path, manifest)
    rows = []
    print(f"Starting {len(layouts)} matched episodes with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(args.config, args.flight, args.coordinator)) as pool:
        for future in as_completed([pool.submit(evaluate_scene, layout, args.output) for layout in layouts]):
            row = future.result()
            old = reference[row["layout_id"]]
            if row["layout_sha256"] != old["layout_sha256"] or row["seed"] != int(old["seed"]):
                raise ValueError(f"Scene differs from the 100M validation: {row['layout_id']}")
            rows.append(row)
            print(f"{len(rows)}/{len(layouts)} {row['layout_id']} {row['success_rate']:.3f} "
                  f"wall={row['wall_seconds']:.1f}s", flush=True)
    evaluation.AGENT_COUNTS = (AGENTS,)
    evaluation.summarize(rows, args.output, layouts, [MODE])
    print(f"Completed {len(rows)} matched episodes", flush=True)


if __name__ == "__main__":
    main()
