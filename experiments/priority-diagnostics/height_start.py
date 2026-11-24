import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch

import diagnose
import diagnose_starts
import evaluate_priority as evaluate
from gym_art.quadrotor_multi.obstacles import layouts

build_standard = layouts.build_layout
HEIGHT_OFFSET = 3.3
MODES = ('simultaneous', 'rows', 'height', 'height_rows', 'climb_then_rows')


def paths_in_both_directions(spec, mode):
    plan = build_standard(spec)
    reverse = np.arange(len(plan.starts)) >= len(plan.starts) // 2
    original_starts = plan.starts.copy()
    plan.starts[reverse] = plan.goals[reverse]
    plan.goals[reverse] = original_starts[reverse]
    if mode in ('height', 'height_rows'):
        plan.starts[reverse, 2] += HEIGHT_OFFSET
        plan.goals[reverse, 2] += HEIGHT_OFFSET
        plan.spec = replace(plan.spec, z_max=float(max(plan.starts[:, 2].max(), plan.goals[:, 2].max())))
    layouts.validate_layout(plan)
    return plan


class TwoWayLaunch(diagnose_starts.FlightStart):
    def schedule(self):
        plan = self.core.layout_plan
        azimuth = np.deg2rad(plan.spec.rotation)
        axis = np.array([-np.sin(azimuth), np.cos(azimuth)])
        forward = plan.starts[:, :2] @ axis
        sign = np.sign((plan.goals[:, :2] - plan.starts[:, :2]) @ axis)
        ticks = np.zeros(len(forward), dtype=int)
        if self.mode in ('rows', 'height_rows', 'climb_then_rows'):
            for direction in (-1, 1):
                group = sign == direction
                progress = direction * forward[group]
                bands = np.quantile(np.unique(np.round(progress, 7)), [1 / 3, 2 / 3])
                rank = 2 - np.digitize(progress, bands)
                ticks[group] = np.rint(.25 * rank / self.core.control_dt).astype(int)
        self.upper = sign < 0
        self.ready_at = None
        return ticks

    def policy_inputs(self, model, observations, states):
        core = self.core
        if self.start_ticks is None:
            self.start_ticks = self.schedule()
        tick = diagnose.trace['metrics'].ticks
        if tick <= round(1.5 / core.control_dt):
            self.snapshots.append((core.pos.copy(), core.vel.copy(), core.priority_neighbors.copy()))
        plan = core.layout_plan
        target = plan.goals.copy()
        if self.mode == 'climb_then_rows':
            at_lane = core.pos[self.upper, 2] >= plan.starts[self.upper, 2] + HEIGHT_OFFSET - .3
            if self.ready_at is None and (at_lane.mean() >= .8 or tick * core.control_dt >= 2.):
                self.ready_at = tick
            if self.ready_at is None:
                target = plan.starts.copy()
                target[self.upper, 2] += HEIGHT_OFFSET
            else:
                waiting = tick - self.ready_at < self.start_ticks
                target[waiting] = plan.starts[waiting]
                en_route = np.linalg.norm(core.pos[:, :2] - plan.goals[:, :2], axis=-1) > 1.2
                target[self.upper & (waiting | en_route), 2] += HEIGHT_OFFSET
        else:
            waiting = tick < self.start_ticks
            target[waiting] = plan.starts[waiting]
        self.yield_ticks += np.count_nonzero(np.any(target != plan.goals, axis=1))
        obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32).clone()
        obs[:, :3] += torch.as_tensor(plan.goals - target, dtype=obs.dtype)
        return model.normalize_obs({'obs': obs})


def trial(seed, mode, output):
    diagnose.trace = dict(signal=np.zeros(11), signal_ticks=0)
    evaluate.SEED = seed
    layouts.build_layout = partial(paths_in_both_directions, mode=mode)
    geometry = next(row for row in evaluate.catalogue() if row['id'] == 'straight_open')
    return evaluate.episode(100, geometry, mode, output / str(seed),
        controller_factory=lambda core: TwoWayLaunch(core, mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[1] / "swarm100-1b"
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'),
            initializer=diagnose_starts.init, initargs=(reference/'config.json',
                reference/'checkpoints/checkpoint_000781254_1000005120.pth')) as pool:
        tasks = [pool.submit(trial, seed, mode, args.output) for seed in (50000, 50001) for mode in MODES]
        for n, f in enumerate(as_completed(tasks), 1):
            r = f.result()
            times = r['first_contact_ticks']['neighbor']
            print(n, len(tasks), r['seed'], r['mode'], r['success_rate'],
                  [sum(0 <= t*r['dt'] <= seconds+1e-9 for t in times) for seconds in (1., 1.5, 3.)],
                  flush=True)
