import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from pathlib import Path
import multiprocessing as mp

import numpy as np

import height_start as height
from height_start import diagnose, diagnose_starts, evaluate, layouts

MODES = ('simultaneous', 'rows', 'height_rows')


def fork_layout(spec, mode):
    plan = height.build_standard(spec)
    left = np.arange(len(plan.starts)) >= len(plan.starts) // 2
    theta = np.deg2rad(plan.spec.rotation)
    across = np.array([np.cos(theta), np.sin(theta)])
    projected = plan.goals[left, :2] @ across
    plan.goals[left, :2] -= (2. * projected[:, None]) * across
    if mode == 'height_rows':
        plan.starts[left, 2] += height.HEIGHT_OFFSET
        plan.goals[left, 2] += height.HEIGHT_OFFSET
        plan.spec = replace(plan.spec, z_max=float(max(plan.starts[:, 2].max(), plan.goals[:, 2].max())))
    layouts.validate_layout(plan)
    return plan


class BranchLaunch(height.TwoWayLaunch):
    def schedule(self):
        theta = np.deg2rad(self.core.layout_plan.spec.rotation)
        progress = self.core.layout_plan.starts[:, :2] @ [-np.sin(theta), np.cos(theta)]
        self.upper = self.core.layout_plan.goals[:, :2] @ [np.cos(theta), np.sin(theta)] < 0.
        self.ready_at = None
        if self.mode == 'simultaneous':
            return np.zeros(len(progress), dtype=int)
        layers = np.unique(np.round(progress, 7))
        rank = 2 - np.digitize(progress, np.quantile(layers, [1 / 3, 2 / 3]))
        return np.rint(rank * .25 / self.core.control_dt).astype(int)


def trial(mode, output):
    diagnose.trace = dict(signal=np.zeros(11), signal_ticks=0)
    evaluate.SEED = 50000
    layouts.build_layout = partial(fork_layout, mode=mode)
    layout = next(row for row in evaluate.catalogue() if row['id'] == 'fork_open')
    return evaluate.episode(100, layout, mode, output,
        controller_factory=lambda core: BranchLaunch(core, mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[1] / "swarm100-1b"
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'),
            initializer=diagnose_starts.init, initargs=(reference/'config.json',
            reference/'checkpoints/checkpoint_000781254_1000005120.pth')) as pool:
        jobs = [pool.submit(trial, mode, args.output) for mode in MODES]
        for i, future in enumerate(as_completed(jobs), 1):
            row = future.result()
            times = row['first_contact_ticks']['neighbor']
            print(i, len(jobs), row['mode'], row['success_rate'],
                  [sum(0 <= t*row['dt'] <= s+1e-9 for t in times) for s in (1., 1.5, 3.)], flush=True)
