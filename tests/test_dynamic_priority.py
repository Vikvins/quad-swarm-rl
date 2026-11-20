from types import SimpleNamespace

import numpy as np
import pytest
import torch

from swarm_rl.dynamic_priority import DynamicPriority, DynamicPriorityConfig, inherit_priority
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.run_layout_suite import make_suite_config


def make_controller(metric='distance'):
    core = SimpleNamespace(
        envs=[SimpleNamespace(goal=np.array([x, 0., 1.]), obs_repr='xyz_vxyz_R_omega_floor')
              for x in (4., -2.)],
        num_agents=2, num_use_neighbor_obs=1, control_dt=.1, collision_threshold=.4,
        pos=np.array([[0., 0., 1.], [1., 0., 1.]]),
        vel=np.array([[1., 0., 0.], [-1., 0., 0.]]),
        priority_neighbors=np.array([[1], [0]]),
    )
    return DynamicPriority(core, DynamicPriorityConfig(metric))


def test_hover_anchor_expires_and_preserves_mission_goal():
    controller = make_controller()
    core = controller.core
    mission_goals = np.array([env.goal.copy() for env in core.envs])
    offsets = controller.goal_offsets()
    np.testing.assert_allclose(offsets, [[0., 0., 0.], [-3., 0., 0.]])
    core.vel[:] *= -1.
    core.pos[1, 0] = 1.1
    np.testing.assert_array_equal(controller.goal_offsets(), offsets)
    np.testing.assert_array_equal(controller.goal_offsets(), 0.)
    np.testing.assert_array_equal([env.goal for env in core.envs], mission_goals)
    assert controller.yield_ticks == 2


def test_no_conflict_with_unobserved_neighbor():
    controller = make_controller()
    # The junior has no edge to the senior in the directed observation graph.
    controller.core.priority_neighbors = np.empty((2, 0), dtype=int)
    np.testing.assert_array_equal(controller.goal_offsets(), 0.)


def test_inheritance_passes_the_greatest_priority_through_a_chain():
    score = np.array([10., 2., 1., 4., 3., 6.])
    effective = inherit_priority(score, np.array([1, 3, 0, 2]), np.array([2, 1, 1, 4]))
    np.testing.assert_array_equal(effective, [10., 10., 10., 4., 10., 6.])
    np.testing.assert_array_equal(score, [10., 2., 1., 4., 3., 6.])


def test_stall_metric_prefers_the_drone_that_stopped_progressing():
    controller = make_controller('stall')
    np.testing.assert_array_equal(controller._priority(np.array([3., 5.]), None), [1, 1])
    np.testing.assert_array_equal(controller._priority(np.array([3., 4.9]), None), [2, 0])
    np.testing.assert_array_equal(controller._priority(np.array([3., 4.88]), None), [3, 1])


def test_critic_scores_original_goal_and_only_actor_goal_channels_change():
    controller = make_controller('critic')
    obs = np.arange(24, dtype=np.float32).reshape(2, 12)
    snapshot = obs.copy()

    class Model:
        def normalize_obs(self, inputs):
            return {'obs': inputs['obs'] * 2.}

        def __call__(self, inputs, states, values_only=False):
            assert values_only
            np.testing.assert_array_equal(inputs['obs'].numpy(), snapshot * 2.)
            return {'values': torch.tensor([2., -3.])}

    result = controller.policy_inputs(Model(), obs, None)['obs'].numpy()
    expected = snapshot.copy()
    expected[0, 0] += 4.
    np.testing.assert_array_equal(result, expected * 2.)
    np.testing.assert_array_equal(obs, snapshot)


@pytest.mark.parametrize('layout', ['random_baseline', 'open_room'])
def test_dynamic_preserves_neighbor_observations(layout):
    cfg = make_suite_config({}, layout, 8, 1., numba=False)
    cfg.quads_neighbor_visible_num = 2
    cfg.quads_neighbor_obs_type = 'pos_vel'
    cfg.quads_use_obstacles = True
    cfg.quads_room_dims = (14., 14., 10.)
    cfg.quads_obs_repr = 'xyz_vxyz_R_omega_floor'
    env = make_quadrotor_env_multi(cfg)
    try:
        core = env.unwrapped
        DynamicPriority(core, DynamicPriorityConfig())
        obs, _ = env.reset()
        indices = np.asarray(core.neighborhood_indices())
        np.testing.assert_array_equal(core.priority_neighbors, indices)
        for i, row in enumerate(indices):
            expected = np.concatenate((core.pos[row] - core.pos[i], core.vel[row] - core.vel[i]), axis=1)
            np.testing.assert_allclose(obs[i, 19:31], expected.ravel(), atol=1e-6)
    finally:
        env.close()
