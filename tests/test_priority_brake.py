from types import SimpleNamespace

import numpy as np
import pytest
import torch

from swarm_rl.priority_brake import BrakeConfig, PriorityBrake, predicts_conflict


def controller(positions, speeds, **parameters):
    count = len(positions)
    core = SimpleNamespace(
        num_agents=count, control_dt=.05, collision_threshold=.2,
        pos=np.column_stack([positions, np.zeros(count), np.ones(count)]),
        vel=np.column_stack([speeds, np.zeros(count), np.zeros(count)]),
        envs=[SimpleNamespace(obs_repr='xyz_vxyz_R_omega_floor', goal=np.array([5., 0., 1.]))
              for _ in positions],
    )
    return PriorityBrake(core, BrakeConfig(**parameters))


def test_lower_priority_drones_see_the_committed_braking_intentions():
    shield = controller([2., 1., 0.], [0., 2., 2.])
    # The junior and middle drone have the same speed. The junior only needs
    # to brake once it accounts for the middle drone yielding to the leader.
    np.testing.assert_array_equal(shield.decide(), [False, True, True])


def test_receding_neighbors_do_not_trigger_braking():
    shield = controller([0., .1], [-1., 1.])
    np.testing.assert_array_equal(shield.decide(), [False, False])


def test_prediction_checks_between_trajectory_samples():
    crossing = np.array([[[-1., 0., 0.], [1., 0., 0.]]])
    parked = np.zeros_like(crossing)
    assert predicts_conflict(crossing, parked, 0, 0, .1)


def test_hold_latches_and_releases_after_conflict_clears():
    shield = controller([0., 1.], [1., -1.], hold=.1)
    np.testing.assert_array_equal(shield.decide(), [False, True])
    shield.core.vel *= -1.
    shield.core.pos[1, 0] = 1.2
    np.testing.assert_array_equal(shield.decide(), [False, True])
    np.testing.assert_array_equal(shield.anchors[1], [1., 0., 1.])
    np.testing.assert_array_equal(shield.decide(), [False, False])


def test_long_wait_triggers_a_bounded_temporary_promotion():
    shield = controller([0., 1.], [1., -1.], max_wait=.1, release=.1)
    for _ in range(2):
        np.testing.assert_array_equal(shield.decide(), [False, True])
    for _ in range(2):
        np.testing.assert_array_equal(shield.decide(), [True, False])
    assert shield.promotion_left == 0
    assert shield.promotions == 1


def test_only_junior_command_and_goal_observation_change():
    shield = controller([0., 1.], [1., -1.])
    obs = np.ones((2, 12), dtype=np.float32)
    frozen_goals = [drone.goal.copy() for drone in shield.core.envs]

    class Model:
        def normalize_obs(self, inputs):
            return inputs

        def __call__(self, inputs, states):
            assert inputs['obs'][1, 0] == pytest.approx(5.)
            return {'new_rnn_states': states + 2}

        def action_distribution(self):
            return SimpleNamespace(means=torch.ones(2, 4))

    states = torch.zeros(2, 3)
    nominal = np.zeros((2, 4))
    actions, next_states, normalized = shield.actions(Model(), obs, states, {}, nominal, states + 1.)
    np.testing.assert_array_equal(actions, [[0.] * 4, [1.] * 4])
    torch.testing.assert_close(next_states, torch.tensor([[1.] * 3, [2.] * 3]))
    np.testing.assert_array_equal(normalized['obs'][:, 3:], obs[:, 3:])
    np.testing.assert_array_equal(obs, 1.)
    np.testing.assert_array_equal(nominal, 0.)
    np.testing.assert_array_equal([drone.goal for drone in shield.core.envs], frozen_goals)
