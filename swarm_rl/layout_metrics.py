"""Evaluation metrics count ALL contacts, including the original grace period."""

import numpy as np

from gym_art.quadrotor_multi.obstacles.layouts import obstacle_clearance


class LayoutMetrics:
    def __init__(self, plan, dt, collision_threshold, goal_radius=.3, goal_hold=.5):
        self.plan = plan
        self.dt = dt
        self.collision_threshold = collision_threshold
        self.goal_radius = goal_radius
        self.goal_hold = goal_hold
        n = plan.spec.num_agents
        self.obstacle_contacts = np.zeros(n, bool)
        self.neighbor_contacts = np.zeros(n, bool)
        self.room_contacts = np.zeros(n, bool)
        self.goal_streak = np.zeros(n)
        self.arrival = np.full(n, np.nan)
        self.portal = np.full(n, np.nan)
        self.path_length = np.zeros(n)
        self.positions = [plan.starts.copy()]
        self.ticks = 0

    def update(self, positions, room_contacts):
        positions = np.asarray(positions).copy()
        self.ticks += 1
        self.path_length += np.linalg.norm(positions - self.positions[-1], axis=1)
        self.positions.append(positions)
        self.obstacle_contacts |= obstacle_clearance(positions[:, :2], self.plan.obstacles,
                                                    self.plan.spec.obstacle_size / 2) <= self.plan.spec.agent_radius
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        self.neighbor_contacts |= np.any(distances <= self.collision_threshold, axis=1)
        self.room_contacts |= np.asarray(room_contacts, bool)
        near = np.linalg.norm(positions - self.plan.goals, axis=1) <= self.goal_radius
        self.goal_streak = np.where(near, self.goal_streak + self.dt, 0.)
        newly_arrived = np.isnan(self.arrival) & (self.goal_streak >= self.goal_hold - 1e-8)
        self.arrival[newly_arrived] = self.ticks * self.dt
        angle = np.deg2rad(self.plan.spec.rotation)
        inverse = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        local = positions[:, :2] @ inverse.T
        spec = self.plan.spec
        if spec.kind == 'corridor':
            crossed = local[:, 1] > spec.corridor_length / 2 + spec.obstacle_size / 2 + spec.agent_radius
        elif spec.kind == 'dead_end':
            crossed = ((local[:, 1] < -spec.dead_end_depth / 2 - spec.obstacle_size / 2 - spec.agent_radius)
                       & (np.abs(local[:, 0]) < spec.width / 2 - spec.agent_radius))
        else:
            crossed = np.zeros(spec.num_agents, bool)
        self.portal[np.isnan(self.portal) & crossed] = self.ticks * self.dt

    def agent_rows(self):
        collided = self.obstacle_contacts | self.neighbor_contacts | self.room_contacts
        final_distance = np.linalg.norm(self.positions[-1] - self.plan.goals, axis=1)
        # A deliberately explicit, simple stall definition: <0.1 m net movement
        # over the last 3 seconds while farther than the goal threshold.
        window = min(len(self.positions) - 1, max(1, int(round(3. / self.dt))))
        moved = np.linalg.norm(self.positions[-1] - self.positions[-1 - window], axis=1)
        long_enough = self.ticks * self.dt >= 3.
        return [dict(agent=i, success=int(np.isfinite(self.arrival[i]) and not collided[i]),
                     reached_goal=int(np.isfinite(self.arrival[i])), collision=int(collided[i]),
                     obstacle_collision=int(self.obstacle_contacts[i]), neighbor_collision=int(self.neighbor_contacts[i]),
                     room_collision=int(self.room_contacts[i]),
                     arrival_time=float(self.arrival[i]) if np.isfinite(self.arrival[i]) else None,
                     portal_time=float(self.portal[i]) if np.isfinite(self.portal[i]) else None,
                     crossed_portal=int(np.isfinite(self.portal[i])),
                     timeout=int(not np.isfinite(self.arrival[i])),
                     stalled=int(long_enough and moved[i] < .1 and final_distance[i] > self.goal_radius),
                     path_length=float(self.path_length[i]), final_distance=float(final_distance[i]))
                for i in range(self.plan.spec.num_agents)]

    def summary(self):
        rows = self.agent_rows()
        result = {key + '_rate': float(np.mean([r[key] for r in rows]))
                  for key in ['success', 'reached_goal', 'collision', 'obstacle_collision',
                              'neighbor_collision', 'room_collision', 'crossed_portal', 'timeout', 'stalled']}
        for key in ['arrival_time', 'portal_time', 'path_length', 'final_distance']:
            values = [r[key] for r in rows if r[key] is not None]
            result['mean_' + key] = float(np.mean(values)) if values else None
        result['duration'] = self.ticks * self.dt
        return result
