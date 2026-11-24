"""Summarize variance across scenes and across noise repeats within each scene."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, variance


def agent_successes(row):
    agents = int(row["agents"])
    selected = agents // 4
    total_successes = round(agents * float(row["agent_success"]))
    selected_successes = round(selected * float(row["selected_agent_success"]))
    return {
        "all": 100 * total_successes / agents,
        "selected": 100 * selected_successes / selected,
        "healthy": 100 * (total_successes - selected_successes) / (agents - selected),
    }


def summarize(episodes, manifest):
    keys = ("agents", "condition", "seed", "repetition")
    data = {}
    for row in episodes:
        key = tuple(row[k] if k == "condition" else int(row[k]) for k in keys)
        if key in data:
            raise ValueError(f"Duplicate episode: {key}")
        data[key] = agent_successes(row)

    scene_seeds = manifest["scene_seeds"]
    repeats = manifest["repeats"]
    conditions = manifest["conditions"]
    expected_keys = {
        (n, mode, seed, repeat)
        for n in manifest["agents"]
        for mode in conditions
        for seed in scene_seeds
        for repeat in repeats
    }
    if data.keys() != expected_keys:
        raise ValueError("Episodes do not match the manifest")

    for n in manifest["agents"]:
        for mode in conditions:
            for group in ("all", "selected", "healthy"):
                scene_rates = []
                within_scene_variances = []
                scene_differences = []
                for seed in scene_seeds:
                    rates = [data[n, mode, seed, r][group] for r in repeats]
                    clean_rates = [data[n, "clean", seed, r][group] for r in repeats]
                    scene_rates.append(mean(rates))
                    within_scene_variances.append(variance(rates))
                    scene_differences.append(mean(rate - clean for rate, clean in zip(rates, clean_rates)))
                yield {
                    "agents": n,
                    "condition": mode,
                    "group": group,
                    "mean_pct": mean(scene_rates),
                    "variance_between_scenes_pp2": variance(scene_rates),
                    "mean_variance_within_scene_pp2": mean(within_scene_variances),
                    "delta_vs_clean_pp": mean(scene_differences),
                    "variance_of_delta_between_scenes_pp2": variance(scene_differences),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    results = parser.parse_args().results
    manifest = json.loads((results / "manifest.json").read_text())
    with (results / "episodes.csv").open(newline="") as source:
        rows = list(summarize(csv.DictReader(source), manifest))
    with (results / "variance.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
