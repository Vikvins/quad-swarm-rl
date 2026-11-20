"""Render exact benchmark geometry and a collision-free witness route, without a model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from gym_art.quadrotor_multi.obstacles.layouts import LayoutSpec, build_layout, reference_route, validate_layout


def plot_layout(plan, output, trajectories=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    for x, y, _ in plan.obstacles:
        ax.add_patch(Circle((x, y), plan.spec.obstacle_size / 2, color='#626c76'))
    route = reference_route(plan)
    ax.plot(route[:, 0], route[:, 1], '--', color='#d8a41f', linewidth=1.5, label='Geometric witness route')
    ax.scatter(*plan.starts[:, :2].T, s=14, c='#147fbc', label='Starts')
    ax.scatter(*plan.goals[:, :2].T, s=16, c='#218447', marker='x', label='Goals')
    if trajectories is not None:
        for i in range(trajectories.shape[1]):
            ax.plot(trajectories[:, i, 0], trajectories[:, i, 1], color='#aa3070', alpha=.2, linewidth=.7)
    hx, hy = plan.spec.room_dims[0] / 2, plan.spec.room_dims[1] / 2
    ax.set(xlim=(-hx, hx), ylim=(-hy, hy), xlabel='x (m)', ylabel='y (m)',
           title=f'{plan.spec.kind}: width={plan.spec.width:g} m, agents={plan.spec.num_agents}, seed={plan.spec.seed}')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/layout_preview'))
    parser.add_argument('--agents', type=int, default=100)
    parser.add_argument('--widths', nargs='+', type=float, default=[.8, 1.2, 1.6])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270], default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for kind in ['corridor', 'dead_end']:
        for width in args.widths:
            plan = build_layout(LayoutSpec(kind, num_agents=args.agents, width=width,
                                           seed=args.seed, rotation=args.rotation))
            stem = f'{kind}_width{width:g}_seed{args.seed}'
            plot_layout(plan, args.output / f'{stem}.png')
            (args.output / f'{stem}.json').write_text(json.dumps(plan.to_dict(), indent=2))
            report.append(dict(layout=kind, width=width, **validate_layout(plan)))
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(f'Validated {len(report)} layouts; previews: {args.output.resolve()}')


if __name__ == '__main__':
    main()
