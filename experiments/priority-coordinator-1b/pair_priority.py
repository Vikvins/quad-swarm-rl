"""A global shortlist of conflict pairs and one learned priority per drone."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from swarm_rl.dynamic_priority import closest_distance_squared


class PairPriority(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair_encoder = nn.Sequential(nn.Linear(18, 64), nn.Tanh())
        self.drone_encoder = nn.Sequential(nn.Linear(73, 32), nn.Tanh())
        self.priority = nn.Linear(32, 1)
        self.value = nn.Linear(32, 1)

    def forward(self, features, pairs):
        batch, agents, _ = features.shape
        endpoints = features[torch.arange(batch, device=features.device)[:, None, None], pairs]
        local_positions = endpoints[..., :3] - endpoints[..., :3].mean(dim=2, keepdim=True)
        endpoints = torch.cat((local_positions, endpoints[..., 3:]), dim=-1)
        directed = torch.stack((endpoints.flatten(-2), endpoints.flip((2,)).flatten(-2)), dim=2)
        messages = self.pair_encoder(directed).reshape(batch, -1, 64)
        recipients = pairs.flatten(1)
        totals = features.new_zeros(batch, agents, 64).scatter_add(
            1, recipients[..., None].expand_as(messages), messages
        )
        counts = features.new_zeros(batch, agents).scatter_add(
            1, recipients, torch.ones_like(recipients, dtype=features.dtype)
        )
        hidden = self.drone_encoder(torch.cat((features, totals / counts.clamp_min(1)[..., None]), dim=-1))
        return Normal(self.priority(hidden).squeeze(-1), .5), self.value(hidden.mean(dim=1)).squeeze(-1)


class PairController:
    """Retain the assigned order until the associated conflicts have cleared."""

    def __init__(self, arenas, agents, dt, collision_threshold, args):
        self.arenas = np.arange(arenas)[:, None]
        self.agent_ids = np.arange(agents)
        self.first, self.second = np.triu_indices(agents, 1)
        self.limit = min(args.pairs, len(self.first))
        self.horizon = args.horizon
        self.refresh = max(1, round(args.refresh / dt))
        self.hold_ticks = max(1, round(.2 / dt))
        self.safe_sq = (1.75 * collision_threshold) ** 2
        self.release_sq = (2 * collision_threshold) ** 2
        self.pairs = np.zeros((arenas, self.limit, 2), dtype=np.int64)
        self.locked_edges = np.zeros((arenas, len(self.first)), dtype=bool)
        self.previous_active = np.zeros((arenas, agents), dtype=bool)
        self.remaining = np.zeros((arenas, agents), dtype=np.int64)
        self.anchors = np.zeros((arenas, agents, 3), dtype=np.float32)
        self.scores = np.zeros((arenas, agents), dtype=np.float32)
        self.ticks = np.zeros(arenas, dtype=np.int64)

    def observe(self, pos, vel, ended):
        self.locked_edges[ended] = False
        self.previous_active[ended] = False
        self.remaining[ended] = 0
        self.ticks[ended] = 0
        relative = pos[:, self.first] - pos[:, self.second]
        relative_velocity = vel[:, self.first] - vel[:, self.second]
        distances = closest_distance_squared(relative, relative_velocity, self.horizon)
        current_sq = np.einsum("...i,...i->...", relative, relative)
        self.locked_edges &= (distances < self.release_sq) | (current_sq < self.release_sq)
        if self.limit == 0:
            inactive = np.zeros_like(self.previous_active)
            return self.pairs, np.zeros_like(self.pairs[..., 0], dtype=bool), inactive

        shortest = np.argpartition(distances, self.limit - 1, axis=1)[:, :self.limit]
        self.pairs = np.stack((self.first[shortest], self.second[shortest]), axis=-1)
        already_selected = self.locked_edges[self.arenas, shortest]
        active = (distances[self.arenas, shortest] < self.safe_sq) | already_selected

        previously_committed = np.zeros_like(self.previous_active)
        arena, edge = np.nonzero(self.locked_edges)
        previously_committed[arena, self.first[edge]] = True
        previously_committed[arena, self.second[edge]] = True
        participating = np.zeros_like(self.previous_active)
        arena, slot = np.nonzero(active)
        participating[arena[:, None], self.pairs[arena, slot]] = True
        refresh = ((self.ticks % self.refresh) == 0) | np.any(active & ~already_selected, axis=1)
        sample = participating & ~previously_committed & refresh[:, None]
        self.locked_edges[self.arenas, shortest] |= active
        self.ticks += 1
        return self.pairs, active, sample

    def offsets(self, scores, sample, pair_active, pos, vel, goals):
        self.scores[sample] = scores[sample]
        arena, selected = np.nonzero(pair_active)
        first, second = self.pairs[arena, selected].T
        first_score, second_score = self.scores[arena, first], self.scores[arena, second]
        first_wins = (first_score > second_score) | ((first_score == second_score) & (first < second))
        lower = np.where(first_wins, second, first)
        yielding = np.zeros_like(self.previous_active)
        yielding[arena, lower] = True

        # A holding drone can still travel. These displacement/speed ratios
        # (seconds) are the 90th percentiles over the first 0.2 s of 429
        # contact-free hold starts in the recorded narrow corridor.
        # Treat the held trajectory conservatively when checking its followers.
        speed = np.linalg.norm(vel, axis=-1)
        drift = np.take((.3165, .2628, .2125, .2269), np.digitize(speed, (.8, 1.5, 2.5)))
        hold_scale = (drift / .2)[..., None]
        active = yielding | (self.remaining > 0)
        first, second = self.pairs.transpose(2, 0, 1)
        first_waits = active[self.arenas, first]
        second_waits = active[self.arenas, second]
        drifts = np.where(active[..., None], hold_scale * vel, vel)
        still_close = closest_distance_squared(
            pos[self.arenas, first] - pos[self.arenas, second],
            drifts[self.arenas, first] - drifts[self.arenas, second], self.horizon
        ) < self.safe_sq
        arena, slot = np.nonzero(still_close & (first_waits ^ second_waits))
        follower = np.where(first_waits[arena, slot], second[arena, slot], first[arena, slot])
        # The same total order applies during the follower check, so two
        # higher-priority drones cannot force each other to wait.
        holder = np.where(first_waits[arena, slot], first[arena, slot], second[arena, slot])
        subordinate = (self.scores[arena, follower] < self.scores[arena, holder]) | (
            (self.scores[arena, follower] == self.scores[arena, holder]) & (follower > holder)
        )
        yielding[arena[subordinate], follower[subordinate]] = True
        started = yielding & ~self.previous_active
        self.anchors[started] = pos[started]
        self.remaining[yielding & (self.remaining == 0)] = self.hold_ticks
        active = self.remaining > 0
        self.remaining[active] -= 1
        self.previous_active = active
        return np.where(active[..., None], goals - self.anchors, 0.)

    def preview_offsets(self, scores, sample, pair_active, pos, vel, goals):
        """Get the next effective goal for value bootstrapping without advancing holds."""
        before = self.scores.copy(), self.anchors.copy(), self.remaining.copy(), self.previous_active.copy()
        try:
            return self.offsets(scores, sample, pair_active, pos, vel, goals)
        finally:
            self.scores, self.anchors, self.remaining, self.previous_active = before


def drone_features(pos, vel, goals):
    return np.concatenate((pos / 5, vel, (goals - pos) / 5), axis=-1)
