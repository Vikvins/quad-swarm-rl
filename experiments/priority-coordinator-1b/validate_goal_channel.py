"""Evaluate the fine-tuned channel on the original message-ablation scenes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path

import evaluate_messages as ablation


MODES = ("messages_real", "messages_zero")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "flight", "peer", "reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    evaluation = ablation.evaluation
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    with args.reference.open(newline="") as source:
        reference = {(row["layout_id"], row["mode"]): row for row in csv.DictReader(source)
                     if row["mode"] in MODES}
    tasks = [(layout, mode) for layout in layouts for mode in MODES]
    if set(reference) != {(layout["id"], mode) for layout, mode in tasks}:
        raise ValueError("Incomplete reference benchmark")
    files = (args.peer, args.config, args.flight, args.reference, Path(__file__), Path(ablation.__file__))
    manifest = dict(agents=100, seed=evaluation.SEED, duration=evaluation.DURATION, modes=list(MODES),
                    files={path.name: ablation.gated.file_hash(path) for path in files})
    path = args.output / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing validation uses different weights")
    evaluation.record_json(path, manifest)
    print(f"Starting {len(tasks)} validation episodes", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=ablation.initialize,
                             initargs=(args.config, args.flight, args.peer)) as pool:
        futures = [pool.submit(ablation.run_case, layout, mode, args.output) for layout, mode in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            previous = reference[row["layout_id"], row["mode"]]
            if row["layout_sha256"] != previous["layout_sha256"] or row["seed"] != int(previous["seed"]):
                raise ValueError("Validation scene differs from the reference")
            rows.append(row)
            print(f"{i}/{len(tasks)} {row['layout_id']} {row['mode']} {row['success_rate']:.3f}", flush=True)
    ablation.MODES = MODES
    ablation.summarize(rows, args.output, layouts)
    print("Validation completed", flush=True)


if __name__ == "__main__":
    main()
