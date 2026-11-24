import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

import diagnose
import evaluate_priority as evaluate


class FlightStart:
    def __init__(self, core, mode):
        self.core = core
        self.mode = mode
        core.track_priority_neighbors = True
        self.yield_ticks = 0
        self.snapshots = []
        self.start_ticks = None
        diagnose.trace['flight_start'] = self

    def schedule(self):
        plan = self.core.layout_plan
        delay = {'probe_base': 0., 'rows_150ms': .15, 'rows_250ms': .25,
                 'shuffled_250ms': .25, 'everyone_500ms': .25}[self.mode]
        angle = np.deg2rad(plan.spec.rotation)
        forward = plan.starts[:, :2] @ [-np.sin(angle), np.cos(angle)]
        layers = np.unique(np.round(forward, 7))
        rank = 2 - np.digitize(forward, np.quantile(layers, [1 / 3, 2 / 3]))
        if self.mode == 'shuffled_250ms':
            rank = np.random.default_rng(50000).permutation(rank)
        if self.mode == 'everyone_500ms':
            rank[:] = 2
        return np.rint(delay * rank / self.core.control_dt).astype(int)

    def policy_inputs(self, model, observations, states):
        core = self.core
        if self.start_ticks is None:
            self.start_ticks = self.schedule()
        tick = diagnose.trace['metrics'].ticks
        if tick <= round(1.5 / core.control_dt):
            self.snapshots.append((core.pos.copy(), core.vel.copy(), core.priority_neighbors.copy()))
        waiting = tick < self.start_ticks
        inputs = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        if waiting.any():
            goals = core.layout_plan.goals
            # While waiting, the flight policy tracks the drone's start XYZ.
            offsets = goals[waiting] - core.layout_plan.starts[waiting]
            inputs = inputs.clone()
            inputs[torch.from_numpy(waiting), :3] += torch.as_tensor(offsets, dtype=inputs.dtype)
            self.yield_ticks += int(waiting.sum())
        return model.normalize_obs({'obs': inputs})


class FlightStartMetrics(diagnose.TracedMetrics):
    def __init__(self, plan, dt, threshold):
        super().__init__(plan, dt, threshold)
        self.pairs = {}
        self.pair_slots = np.triu_indices(len(plan.starts), 1)

    def update(self, positions, room_contacts):
        tick = self.ticks + 1
        if tick * self.dt <= 1.5 + 1e-9:
            i, j = self.pair_slots
            close = np.linalg.norm(positions[i] - positions[j], axis=1) <= self.collision_threshold
            for left, right in zip(i[close], j[close]):
                self.pairs.setdefault((left, right), tick)
        super().update(positions, room_contacts)

    def summary(self):
        out = super().summary()
        test = diagnose.trace['flight_start']
        snaps = test.snapshots
        out['wait_schedule'] = {int(t): int((test.start_ticks == t).sum()) for t in set(test.start_ticks)}
        out['snapshots_kinematics'] = {}
        direction = self.plan.goals.mean(axis=0) - self.plan.starts.mean(axis=0)
        direction[2] = 0
        direction /= np.linalg.norm(direction)
        for second in (0., .25, .4, .5, 1.):
            _, vel, _ = snaps[round(second / self.dt)]
            out['snapshots_kinematics'][second] = dict(
                median_speed=float(np.median(np.linalg.norm(vel, axis=1))),
                median_forward_velocity=float(np.median(vel @ direction)),
            )
        out['early_pairs'] = []
        for (i,j), tick in sorted(self.pairs.items(), key=lambda item:item[1]):
            def observation(lead):
                before = max(0, tick - max(1, round(lead / self.dt)))
                pos, vel, neighbors = snaps[before]
                delta = pos[j] - pos[i]
                dist = np.linalg.norm(delta)
                return dict(visible=[bool(j in neighbors[i]), bool(i in neighbors[j])],
                    distance=float(dist), separation_rate=float(np.dot(delta,vel[j]-vel[i])/dist),
                    positions=pos[[i,j]].tolist(), velocities=vel[[i,j]].tolist())
            out['early_pairs'].append(dict(ids=[int(i),int(j)], tick=tick,
                before_10ms=observation(.01), before_100ms=observation(.10),
                goals=self.plan.goals[[i,j]].tolist()))
        return out


def init(config, flight, downwash=True):
    evaluate.initialize(config, flight, None)
    evaluate._CONFIG['quads_use_downwash'] = downwash
    evaluate.LayoutMetrics = FlightStartMetrics


def run(layout, mode, output):
    diagnose.trace = dict(signal=np.zeros(11), signal_ticks=0)
    return evaluate.episode(100, layout, mode, output,
        controller_factory=lambda core: FlightStart(core, mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-downwash", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1] / "swarm100-1b"
    layout = next(layout for layout in evaluate.catalogue() if layout['id'] == 'narrow_open')
    modes = ('probe_base','rows_150ms','rows_250ms','shuffled_250ms','everyone_500ms')
    if args.disable_downwash:
        modes = ("probe_base",)
    with ProcessPoolExecutor(max_workers=len(modes), mp_context=mp.get_context('spawn'),
            initializer=init, initargs=(repo/'config.json',
                repo/'checkpoints/checkpoint_000781254_1000005120.pth',
                not args.disable_downwash)) as pool:
        jobs = [pool.submit(run, layout, mode, args.output) for mode in modes]
        for future in as_completed(jobs):
            row = future.result()
            print(row['mode'], 'safe', row['success_rate'], flush=True)
