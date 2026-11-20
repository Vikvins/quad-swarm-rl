"""Side-by-side comparison of layout-suite runs made with different --static_priority modes.

Each mode is run into its own folder by swarm_rl.run_layout_suite. This script reads episodes.csv from every
folder, keeps only episodes present in all runs (same layout, number of drones and seed, so the starts, goals
and walls are identical), and prints the per-mode means and paired differences with
95% bootstrap intervals. Episodes are resampled by seed, independently for each layout and swarm size.

Example:
    python scripts/compare_static_priority.py --runs off=results/suite_off mask=results/suite_mask \
        --output results/priority_comparison.csv
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ['success_rate', 'collision_rate', 'neighbor_collision_rate', 'obstacle_collision_rate',
           'stalled_rate', 'timeout_rate', 'mean_arrival_time', 'brake_tick_rate', 'priority_releases']
KEYS = ['agents', 'layout_id', 'seed']


def bootstrap_delta(delta, rng, samples):
    delta = np.asarray(delta)
    delta = delta[np.isfinite(delta)]
    count = len(delta)
    if count == 0:
        return np.nan, np.nan, np.nan, 0
    if count == 1:
        return float(delta[0]), np.nan, np.nan, 1
    indices = rng.integers(count, size=(samples, count))
    lower, upper = np.quantile(delta[indices].mean(axis=1), [.025, .975])
    return float(delta.mean()), lower, upper, count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='name=folder, e.g. off=results/suite_off')
    parser.add_argument('--output', type=Path, help='optional CSV with the comparison table')
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--bootstrap_seed', type=int, default=0)
    args = parser.parse_args()
    if len(args.runs) < 2 or args.bootstrap_samples < 1:
        parser.error('Provide at least two runs and a positive bootstrap sample count')

    frames = []
    for item in args.runs:
        name, folder = item.split('=', 1)
        data = pd.read_csv(Path(folder) / 'episodes.csv')
        if data.empty or data.duplicated(KEYS).any():
            parser.error(f'{folder}/episodes.csv must contain one row per episode key')
        data['run'] = name
        frames.append(data)
    names = [f.run.iloc[0] for f in frames]
    if len(set(names)) != len(names):
        parser.error('Run names must be unique')

    # paired comparison: only episodes that exist in every run
    common = frames[0][KEYS]
    for f in frames[1:]:
        common = common.merge(f[KEYS], on=KEYS)
    if common.empty:
        parser.error('The runs have no common episodes')
    data = pd.concat([f.merge(common, on=KEYS) for f in frames])
    if 'layout_sha256' in data:
        if data.groupby(KEYS)['layout_sha256'].nunique().gt(1).any():
            parser.error('At least one paired seed has different starts, goals or obstacles')
        if data['layout_sha256'].isna().any():
            print('Some episodes have no layout fingerprint. Their geometry could not be verified.')
    else:
        print('Layout fingerprints are unavailable. Episodes are matched by seed only.')
    metrics = [m for m in METRICS if m in data.columns]

    table = data.groupby(['agents', 'layout_id', 'run'])[metrics].mean().unstack('run')
    table = table.reindex(columns=pd.MultiIndex.from_product([metrics, names]))
    episodes_by_mode = data.set_index(KEYS + ['run'])[metrics].unstack('run').sort_index()
    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = {}
    for metric in metrics:
        for baseline, candidate in combinations(names, 2):
            comparison = f'{candidate}-{baseline}'
            delta = episodes_by_mode[(metric, candidate)] - episodes_by_mode[(metric, baseline)]
            intervals = delta.groupby(level=['agents', 'layout_id']).apply(
                lambda values: bootstrap_delta(values, rng, args.bootstrap_samples))
            for stat, suffix in enumerate(('', '_lo95', '_hi95', '_n')):
                comparisons[(metric, comparison + suffix)] = intervals.map(lambda result: result[stat])
    table = pd.concat([table, pd.DataFrame(comparisons)], axis=1)
    table = table.sort_index(axis=1, level=0, sort_remaining=False)
    table[('episodes', '')] = data[data.run == names[0]].groupby(['agents', 'layout_id']).size()

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.max_rows', 200)
    main_cols = [c for c in table.columns if c[0] in ('success_rate', 'neighbor_collision_rate', 'episodes')
                 and not c[1].endswith('_n')]
    print(table[main_cols].round(3))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.output)
        print(f'\nfull table: {args.output}')


if __name__ == '__main__':
    main()
