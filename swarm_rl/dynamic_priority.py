"""Temporary goal overrides for frozen-policy evaluation."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class DynamicPriorityConfig:
    metric: str = 'distance'
    lookahead: float = .5
    hold: float = .2
    conflict_scale: float = 1.75

    def __post_init__(self):
        if self.metric not in ('distance', 'stall', 'critic'):
            raise ValueError(f'Unknown dynamic priority metric: {self.metric}')
        if not all(math.isfinite(x) and x > 0 for x in (self.lookahead, self.hold, self.conflict_scale)):
            raise ValueError('Dynamic priority thresholds must be finite and positive')


def add_dynamic_priority_args(parser):
    parser.add_argument('--dynamic_priority_metric', choices=('distance', 'stall', 'critic'), default='distance')
    parser.add_argument('--dynamic_priority_lookahead', type=float, default=.5, help='conflict prediction horizon (s)')
    parser.add_argument('--dynamic_priority_hold', type=float, default=.2, help='duration of each hover goal (s)')
    parser.add_argument('--dynamic_priority_conflict_scale', type=float, default=1.75,
                        help='conflict distance as a multiple of the simulator collision threshold')


def dynamic_priority_config(args):
    return DynamicPriorityConfig(args.dynamic_priority_metric, args.dynamic_priority_lookahead,
                                 args.dynamic_priority_hold, args.dynamic_priority_conflict_scale)


def closest_distance_squared(position, velocity, horizon):
    speed_squared = np.sum(velocity * velocity, axis=-1)
    approach = -np.sum(position * velocity, axis=-1)
    time = np.clip(approach / np.maximum(speed_squared, 1e-12), 0., horizon)
    offset = position + time[..., None] * velocity
    return np.sum(offset * offset, axis=-1)


def inherit_priority(score, sources, blockers):
    """Pass the greatest urgency along the dependency DAG in one traversal."""
    if not len(sources):
        return score
    n = len(score)
    outgoing = [[] for _ in range(n)]
    incoming = np.bincount(blockers, minlength=n)
    for source, blocker in zip(sources, blockers):
        outgoing[source].append(blocker)
    queue = deque(np.flatnonzero((incoming == 0) & (np.bincount(sources, minlength=n) > 0)))
    effective = score.copy()
    while queue:
        source = queue.popleft()
        for blocker in outgoing[source]:
            effective[blocker] = max(effective[blocker], effective[source])
            incoming[blocker] -= 1
            if incoming[blocker] == 0:
                queue.append(blocker)
    return effective


class DynamicPriority:
    """One controller per episode; uses the exact neighbor slots already built by the simulator."""

    def __init__(self, core, config):
        if core.num_use_neighbor_obs < 1:
            raise ValueError('Dynamic priority requires neighbor observations')
        if any(not env.obs_repr.startswith('xyz_') for env in core.envs):
            raise ValueError('Dynamic priority requires the goal-relative xyz observation prefix')
        self.core = core
        self.config = config
        self.ids = np.arange(core.num_agents)[:, None]
        self.hold_ticks = math.ceil(config.hold / core.control_dt)
        self.threshold_squared = (core.collision_threshold * config.conflict_scale) ** 2
        self.best_distance = None
        self.stall_ticks = np.zeros(core.num_agents, dtype=int)
        self.remaining = np.zeros(core.num_agents, dtype=int)
        self.anchors = np.zeros_like(core.pos)
        self.yield_ticks = 0
        self.inherited_ticks = 0
        core.track_priority_neighbors = True

    def _priority(self, distance, critic_values):
        if self.config.metric == 'distance':
            return distance
        if self.config.metric == 'critic':
            score = -np.asarray(critic_values).reshape(-1)
            if score.shape != distance.shape or not np.isfinite(score).all():
                raise ValueError('Expected one finite critic value per drone')
            return score

        # Count consecutive ticks without at least 5 cm of progress toward the mission goal.
        if self.best_distance is None:
            self.best_distance = distance.copy()
        progressing = (distance <= self.best_distance - .05) | (distance <= .3)
        self.best_distance[progressing] = distance[progressing]
        self.stall_ticks = np.where(progressing, 0, self.stall_ticks + 1)
        return self.stall_ticks

    def goal_offsets(self, critic_values=None):
        core = self.core
        neighbors = core.priority_neighbors
        pos, vel = core.pos, core.vel
        goals = np.asarray([env.goal for env in core.envs])
        score = self._priority(np.linalg.norm(pos - goals, axis=1), critic_values)

        if not neighbors.size:
            return pos - pos
        relative_pos = pos[neighbors] - pos[:, None, :]
        relative_vel = vel[neighbors] - vel[:, None, :]
        horizon = self.config.lookahead
        conflict = closest_distance_squared(relative_pos, relative_vel, horizon) < self.threshold_squared

        base = score[:, None]
        other = score[neighbors]
        base_senior = (other > base) | ((other == base) & (neighbors < self.ids))
        yielding = np.any(conflict & base_senior, axis=1) | (self.remaining > 0)

        # A nearby junior approaching the yielding drone's hover point can block that
        # maneuver. Pass urgency down to it, including along chains of yielding drones.
        # Edges descend in base order, which makes the graph acyclic even on ties.
        hold_points = np.where(self.remaining[:, None] > 0, self.anchors, pos)
        blocks_hover = closest_distance_squared(pos[neighbors] - hold_points[:, None, :],
                                                vel[neighbors], horizon) < self.threshold_squared
        source, slot = np.nonzero(yielding[:, None] & ~base_senior & blocks_hover)
        effective = inherit_priority(score, source, neighbors[source, slot])
        self.inherited_ticks += int(np.count_nonzero(effective > score))

        own, other = effective[:, None], effective[neighbors]
        senior = (other > own) | ((other == own) & base_senior)
        start = np.any(conflict & senior, axis=1) & (self.remaining == 0)
        self.anchors[start] = pos[start]
        self.remaining[start] = self.hold_ticks
        active = self.remaining > 0
        self.yield_ticks += int(np.count_nonzero(active))
        self.remaining[active] -= 1
        return np.where(active[:, None], goals - self.anchors, 0.)

    def policy_inputs(self, model, obs, states):
        """Score the original goal with the frozen critic before substituting actor goals."""
        inputs = {'obs': torch.as_tensor(np.asarray(obs), dtype=torch.float32)}
        values = None
        if self.config.metric == 'critic':
            original = model.normalize_obs({'obs': inputs['obs'].clone()})
            values = model(original, states, values_only=True)['values'].detach().cpu().numpy()
        offsets = self.goal_offsets(values)
        # The first three channels are measured position minus goal. Adding
        # mission_goal - hover_goal keeps the same sensor noise on position.
        inputs['obs'] = inputs['obs'].clone()
        inputs['obs'][:, :3] += torch.as_tensor(offsets, dtype=inputs['obs'].dtype)
        return model.normalize_obs(inputs)
