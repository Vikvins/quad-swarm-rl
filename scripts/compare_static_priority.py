"""Side-by-side comparison of layout-suite runs made with different --static_priority modes.

Each mode is run into its own folder by swarm_rl.run_layout_suite. This script reads episodes.csv from every
folder, keeps only episodes present in all runs (same layout, number of drones and seed, so the starts, goals
and walls are identical), and prints the mean of the main metrics per layout and mode, plus the difference
of every mode from the first one.

Example:
    python scripts/compare_static_priority.py --runs off=results/suite_off mask=results/suite_mask \
        --output results/priority_comparison.csv
"""
import argparse
from pathlib import Path

import pandas as pd

METRICS = ['success_rate', 'collision_rate', 'neighbor_collision_rate', 'obstacle_collision_rate',
           'stalled_rate', 'timeout_rate', 'mean_arrival_time']
KEYS = ['agents', 'layout_id', 'seed']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='name=folder, e.g. off=results/suite_off')
    parser.add_argument('--output', type=Path, help='optional CSV with the comparison table')
    args = parser.parse_args()

    frames = []
    for item in args.runs:
        name, folder = item.split('=', 1)
        data = pd.read_csv(Path(folder) / 'episodes.csv')
        data['run'] = name
        frames.append(data)
    names = [f.run.iloc[0] for f in frames]

    # paired comparison: only episodes that exist in every run
    common = frames[0][KEYS]
    for f in frames[1:]:
        common = common.merge(f[KEYS], on=KEYS)
    data = pd.concat([f.merge(common, on=KEYS) for f in frames])
    metrics = [m for m in METRICS if m in data.columns]

    table = data.groupby(['agents', 'layout_id', 'run'])[metrics].mean().unstack('run')
    table = table.reindex(columns=pd.MultiIndex.from_product([metrics, names]))
    for metric in metrics:
        for name in names[1:]:
            table[(metric, f'{name}-{names[0]}')] = table[(metric, name)] - table[(metric, names[0])]
    table = table.sort_index(axis=1, level=0, sort_remaining=False)
    episodes = data[data.run == names[0]].groupby(['agents', 'layout_id']).size().rename(('episodes', ''))
    table = table.join(episodes)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.max_rows', 200)
    main_cols = [c for c in table.columns if c[0] in ('success_rate', 'collision_rate', 'mean_arrival_time', 'episodes')]
    print(table[main_cols].round(3))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.output)
        print(f'\nfull table: {args.output}')


if __name__ == '__main__':
    main()
