"""Compare the same physical braking filter with fixed and learned priorities."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

import diagnose
import evaluate_priority as evaluation
from priority_brake import BrakeConfig, PriorityBrake, resolve_braking
from train_priority import PriorityPolicy


MODES = ("static_brake", "ranked_brake", "gated_peer")
RANK_REFRESH_SECONDS = 0.1
INTENT_SPEED = 0.8


class ChannelRankedBrake(PriorityBrake):
    def __init__(self, core, config):
        super().__init__(core, config)
        self.order = self.ids
        self.policy = None
        self.rank_refresh = max(1, round(RANK_REFRESH_SECONDS / core.control_dt))
        self.tick = 0

    def score(self, obs):
        features = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        if self.policy is None:
            checkpoint = evaluation._PEER_CHECKPOINT
            gain = checkpoint["arguments"].get("peer_message_gain", 1.0)
            self.policy = PriorityPolicy("peer", features.shape[-1], self.core.num_use_neighbor_obs, gain)
            self.policy.load_state_dict(checkpoint["policy"], strict=True)
            self.policy.eval()
        hidden = self.policy.encode(features)
        bids = self.policy.bid(hidden).squeeze(-1)
        if self.tick % self.rank_refresh == 0:
            self.order = np.lexsort((self.ids, -bids.numpy()))
        self.tick += 1
        return hidden, bids

    def priority_order(self):
        return self.order

    def actions(self, model, obs, states, normalized, nominal_actions, nominal_states):
        self.prepare(obs)
        return super().actions(model, obs, states, normalized, nominal_actions, nominal_states)

    def prepare(self, obs):
        if self.tick % self.rank_refresh == 0:
            self.score(obs)
        else:
            self.tick += 1


class ConflictGatedPeerBrake(ChannelRankedBrake):
    def __init__(self, core, config):
        super().__init__(core, config)
        core.track_priority_neighbors = True
        self.no_latched_hold = np.zeros(core.num_agents, dtype=bool)
        self.intent_times = self.nominal_times - self.brake_times
        self.horizon = config.horizon

    def prepare(self, obs):
        hidden, bids = self.score(obs)
        neighbors = torch.as_tensor(self.core.priority_neighbors, dtype=torch.long)
        decision = self.policy.wait_distribution(hidden, bids, neighbors, bids.new_ones((1,)))
        self.requests = decision.logits.numpy() > 0

    def decide(self):
        pos, vel = self.core.pos, self.core.vel
        goals = np.asarray([drone.goal for drone in self.core.envs])
        delta = goals - pos
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        intent = delta / np.maximum(distance, 1e-6)
        intent *= np.minimum(distance / self.horizon, INTENT_SPEED)
        intent *= np.maximum(1.0 - speed / INTENT_SPEED, 0.0)
        nominal = pos[:, None, :] + vel[:, None, :] * self.nominal_times + intent[:, None, :] * self.intent_times
        stopped = pos[:, None, :] + vel[:, None, :] * self.brake_times
        order = self.order_with_promotion()
        braking = resolve_braking(nominal, stopped, order, self.no_latched_hold, self.threshold_squared, self.requests)
        started = braking & (self.waited == 0)
        self.anchors[started] = pos[started]
        self.waited = np.where(braking, self.waited + 1, 0)
        self.brake_ticks += int(braking.sum())
        return braking


def run(agents, layout, mode, output):
    diagnose.trace = dict(signal=np.zeros(11), signal_ticks=0)
    controller = {"ranked_brake": ChannelRankedBrake, "gated_peer": ConflictGatedPeerBrake}.get(mode)
    if controller is None:
        return evaluation.episode(agents, layout, mode, output)
    return evaluation.episode(agents, layout, mode, output,
                              brake_factory=lambda core: controller(core, BrakeConfig()))


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "flight", "peer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    layouts = [layout for layout in evaluation.catalogue() if layout["paired_with"]]
    if len(layouts) != 16:
        raise ValueError("Expected eight corridors and eight paired dead ends")
    manifest = dict(agents=args.agents, seed=evaluation.SEED, duration=evaluation.DURATION,
                    modes=list(MODES), layouts=[l["id"] for l in layouts], brake=vars(BrakeConfig()),
                    rank_refresh_seconds=RANK_REFRESH_SECONDS, gated_intent_speed=INTENT_SPEED,
                    files={p.name: file_hash(p) for p in (args.config, args.flight, args.peer,
                           Path(__file__), Path(evaluation.__file__), Path(diagnose.__file__),
                           Path(__file__).with_name("priority_brake.py"))})
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to mix different validation configurations")
    evaluation.record_json(path, manifest)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=diagnose.init, initargs=(args.config, args.flight, args.peer)) as pool:
        jobs = [pool.submit(run, args.agents, layout, mode, args.output) for layout in layouts for mode in MODES]
        for i, future in enumerate(as_completed(jobs), 1):
            row = future.result()
            rows.append(row)
            print(i, len(jobs), row["layout_id"], row["mode"], row["success_rate"], flush=True)

    evaluation.AGENT_COUNTS = (args.agents,)
    metrics = ("first_contact_ticks", "first_hold_ticks", "signal", "signal_ticks",
               "nearest_start", "start_span_xyz", "start_clearance", "collision_threshold", "dt")
    evaluation.summarize([{k: v for k, v in r.items() if k not in metrics} for r in rows],
                         args.output, layouts, MODES)
    early = []
    for row in sorted(rows, key=lambda r: (r["layout_id"], r["mode"])):
        hit = row["first_contact_ticks"]["neighbor"]
        early.append(dict(layout_id=row["layout_id"], kind=row["kind"], mode=row["mode"],
                          neighbor_contacts_by_1p5s=sum(0 <= t * row["dt"] <= 1.5 + 1e-9 for t in hit),
                          neighbor_contacts_after_3s=sum(t * row["dt"] > 3 + 1e-9 for t in hit)))
    evaluation.write_csv(args.output / "early-contacts.csv", early)
    print("48 episodes verified on matched scenes.", flush=True)


if __name__ == "__main__":
    main()
