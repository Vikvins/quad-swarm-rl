"""All-pairs static-priority braking for frozen-policy simulation."""
from dataclasses import dataclass
import math

from numba import njit
import numpy as np
import torch


@dataclass(frozen=True)
class BrakeConfig:
    horizon: float = .65
    distance_scale: float = 1.8
    stopping_time: float = .25
    hold: float = .20
    max_wait: float = 2.
    release: float = .5

    def __post_init__(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Brake parameter {key} must be finite and positive')


def add_brake_args(parser):
    defaults = BrakeConfig()
    for name, help_text in (
        ('horizon', 'trajectory prediction horizon (s)'),
        ('distance_scale', 'safety distance divided by the simulator collision threshold'),
        ('stopping_time', 'assumed time for a braking drone to stop (s)'),
        ('hold', 'minimum time to retain a hover goal (s)'),
        ('max_wait', 'continuous waiting before temporary promotion becomes eligible (s)'),
        ('release', 'duration of temporary promotion (s)'),
    ):
        parser.add_argument(f'--brake_{name}', type=float, default=getattr(defaults, name), help=help_text)


def brake_config_from_args(args):
    return BrakeConfig(**{name: getattr(args, f'brake_{name}') for name in vars(BrakeConfig())})


@njit(cache=True)
def predicts_conflict(nominal, accepted, i, j, threshold_squared):
    """Minimum separation over every linear segment, with no sampling gaps."""
    dx = nominal[i, 0, 0] - accepted[j, 0, 0]
    dy = nominal[i, 0, 1] - accepted[j, 0, 1]
    dz = nominal[i, 0, 2] - accepted[j, 0, 2]
    # A close pair that maintains separation or moves apart does not start a new hold.
    limit = min(threshold_squared, dx * dx + dy * dy + dz * dz - 1e-8)
    for tick in range(1, nominal.shape[1]):
        nx = nominal[i, tick, 0] - accepted[j, tick, 0]
        ny = nominal[i, tick, 1] - accepted[j, tick, 1]
        nz = nominal[i, tick, 2] - accepted[j, tick, 2]
        ux, uy, uz = nx - dx, ny - dy, nz - dz
        speed_squared = ux * ux + uy * uy + uz * uz
        t = min(1., max(0., -(dx * ux + dy * uy + dz * uz) / max(speed_squared, 1e-12)))
        x, y, z = dx + t * ux, dy + t * uy, dz + t * uz
        if x * x + y * y + z * z < limit:
            return True
        dx, dy, dz = nx, ny, nz
    return False


@njit(cache=True)
def resolve_braking(nominal, stopped, order, held, threshold_squared, eligible=None):
    """Process juniors against the accepted trajectories of all senior drones."""
    accepted = nominal.copy()
    brakes = held.copy()
    for rank, drone in enumerate(order):
        if not brakes[drone] and (eligible is None or eligible[drone]):
            for senior in order[:rank]:
                if predicts_conflict(nominal, accepted, drone, senior, threshold_squared):
                    brakes[drone] = True
                    break
        if brakes[drone]:
            accepted[drone] = stopped[drone]
    return brakes


class PriorityBrake:
    def __init__(self, core, config):
        if any(not drone.obs_repr.startswith('xyz_') for drone in core.envs):
            raise ValueError('Brake mode requires goal-relative xyz observation channels')
        self.core = core
        self.ids = np.arange(core.num_agents)
        self.remaining = np.zeros(core.num_agents, dtype=np.int64)
        self.waited = np.zeros_like(self.remaining)
        self.anchors = np.zeros_like(core.pos)
        self.hold_ticks = math.ceil(config.hold / core.control_dt)
        self.wait_limit = math.ceil(config.max_wait / core.control_dt)
        self.release_ticks = math.ceil(config.release / core.control_dt)
        self.threshold_squared = (core.collision_threshold * config.distance_scale) ** 2
        timeline = np.unique(np.r_[np.linspace(0., config.horizon, 9), min(config.stopping_time, config.horizon)])
        self.nominal_times = timeline[None, :, None]
        # Integral of a velocity that ramps linearly to zero over stopping_time.
        ramp = np.minimum(timeline, config.stopping_time)
        self.brake_times = (ramp - .5 * ramp ** 2 / config.stopping_time)[None, :, None]
        self.promoted = -1
        self.promotion_left = 0
        self.promotions = 0
        self.brake_ticks = 0

    def priority_order(self):
        return self.ids

    def order_with_promotion(self):
        order = self.priority_order()
        if self.promotion_left == 0:
            self.promoted = -1
            candidate = int(np.argmax(self.waited))
            if self.waited[candidate] >= self.wait_limit:
                self.promoted = candidate
                self.promotion_left = self.release_ticks
                self.promotions += 1
        if self.promoted >= 0:
            order = np.r_[self.promoted, order[order != self.promoted]]
            self.remaining[self.promoted] = 0
            self.promotion_left -= 1

        return order

    def decide(self):
        pos, vel = self.core.pos, self.core.vel
        nominal = pos[:, None, :] + vel[:, None, :] * self.nominal_times
        stopped = pos[:, None, :] + vel[:, None, :] * self.brake_times
        order = self.order_with_promotion()
        held = self.remaining > 0
        braking = resolve_braking(nominal, stopped, order, held, self.threshold_squared)
        new_hold = braking & ~held
        # Hold one fixed position at a time. Renew the timer while the conflict
        # persists, but keep the same anchor until the drone is released.
        started = braking & (self.waited == 0)
        self.anchors[started] = pos[started]
        self.remaining[new_hold] = self.hold_ticks
        self.remaining[braking] -= 1
        self.waited = np.where(braking, self.waited + 1, 0)
        self.brake_ticks += int(braking.sum())
        return braking

    def actions(self, model, obs, states, normalized, nominal_actions, nominal_states):
        braking = self.decide()
        if not braking.any():
            return nominal_actions, nominal_states, normalized
        goals = np.asarray([drone.goal for drone in self.core.envs])
        changed = torch.as_tensor(np.asarray(obs), dtype=torch.float32).clone()
        changed[braking, :3] += torch.as_tensor(goals[braking] - self.anchors[braking], dtype=changed.dtype)
        normalized = model.normalize_obs({'obs': changed})
        outputs = model(normalized, states)
        actions = nominal_actions.copy()
        actions[braking] = model.action_distribution().means.detach().cpu().numpy()[braking]
        next_states = nominal_states.clone()
        next_states[braking] = outputs['new_rnn_states'][braking]
        return actions, next_states, normalized
