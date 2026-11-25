import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

import evaluate_priority as evaluate

PeerPriority = evaluate.PeerPriority
LayoutMetrics = evaluate.LayoutMetrics
trace = None


class TracedPeer(PeerPriority):
    def policy_inputs(self, model, observations, states):
        core = self.core
        features = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        hidden = self.policy.encode(features)
        raw = self.policy.bid(hidden)[:, 0]
        neighbors = torch.as_tensor(core.neighborhood_indices(), dtype=torch.long)
        signals = torch.cat((raw[:, None], raw[neighbors]), dim=-1).tanh() * self.policy.message_gain
        n = len(raw)
        flag = features.new_ones(n, 1)
        zeros = torch.zeros_like(signals)

        def decide(messages, flag):
            return self.policy.decide(torch.cat((hidden, messages, flag), dim=-1)).squeeze(-1)

        on = decide(signals, flag)
        zero = decide(zeros, flag)
        off = decide(zeros, torch.zeros_like(flag))
        swap = decide(torch.cat((signals[:, :1], signals.roll(1, 0)[:, 1:]), dim=-1), flag)
        trace["signal"] += np.array([
            n, (on > 0).sum().item(), (off > 0).sum().item(), ((on > 0) != (off > 0)).sum().item(),
            ((on > 0) != (zero > 0)).sum().item(), ((on > 0) != (swap > 0)).sum().item(),
            (on - zero).abs().sum().item(), (zero - off).abs().sum().item(),
            (raw.abs() > 3).sum().item(), raw.std(unbiased=False).item(), raw.abs().mean().item(),
        ])
        trace["signal_ticks"] += 1
        ticks = trace["metrics"].ticks
        waits = (on if self.channel_enabled else off).numpy() > 0
        first_hold = trace["first_hold"]
        first_hold[waits & (first_hold < 0)] = ticks
        return super().policy_inputs(model, observations, states)


class TracedMetrics(LayoutMetrics):
    def __init__(self, plan, dt, threshold):
        super().__init__(plan, dt, threshold)
        trace["metrics"] = self
        count = len(plan.starts)
        trace["first_hold"] = np.full(count, -1, dtype=int)
        self.contact_times = {k: np.full(count, -1, dtype=int) for k in ('neighbor', 'obstacle', 'room')}

    def log_contacts(self):
        for key, first in self.contact_times.items():
            current = getattr(self, key + "_contacts")
            first[current & (first < 0)] = self.ticks

    def update(self, positions, room_contacts):
        self.log_contacts()
        super().update(positions, room_contacts)
        self.log_contacts()

    def summary(self):
        row = super().summary()
        plan = self.plan
        distances = np.linalg.norm(plan.starts[:, None] - plan.starts[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        row["nearest_start"] = distances.min(axis=1).tolist()
        row["start_span_xyz"] = np.ptp(plan.starts, axis=0).tolist()
        row["start_clearance"] = evaluate.obstacle_clearance(
            plan.starts[:, :2], plan.obstacles, plan.spec.obstacle_size / 2).tolist()
        row["collision_threshold"] = float(self.collision_threshold)
        row["dt"] = float(self.dt)
        row["first_contact_ticks"] = {k: t.tolist() for k,t in self.contact_times.items()}
        row["first_hold_ticks"] = trace["first_hold"].tolist()
        row["signal_ticks"] = trace["signal_ticks"]
        row["signal"] = trace["signal"].tolist()
        return row


def init(config, flight, peer):
    evaluate.initialize(config, flight, None, peer)
    evaluate.LayoutMetrics = TracedMetrics
    evaluate.PeerPriority = TracedPeer


def run(n, layout, mode, output):
    global trace
    trace = dict(signal=np.zeros(11), signal_ticks=0)
    return evaluate.episode(n, layout, mode, output)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    base = repo / "swarm100-1b"
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--agents", type=int, nargs="+", default=[100])
    parser.add_argument("--modes", nargs="+", default=["baseline", "peer_on", "peer_off", "static_brake"])
    parser.add_argument("--config", type=Path, default=base / "config.json")
    parser.add_argument("--flight", type=Path,
                        default=base / "checkpoints/checkpoint_000781254_1000005120.pth")
    parser.add_argument("--peer", type=Path, default=repo / "priority-channel-1b/priority.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layouts = [layout for layout in evaluate.catalogue() if layout["paired_with"]]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=init, initargs=(args.config, args.flight, args.peer)) as pool:
        tasks = [pool.submit(run, n, layout, mode, args.output)
                 for n in args.agents for layout in layouts for mode in args.modes]
        for i, f in enumerate(as_completed(tasks), 1):
            row = f.result()
            print(i, len(tasks), row["agents"], row["layout_id"], row["mode"], row["success_rate"], flush=True)
