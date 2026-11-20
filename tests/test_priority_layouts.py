from dataclasses import replace
import os
import numpy as np
import pytest

from gym_art.quadrotor_multi.obstacles.layouts import (
    LayoutSpec, build_layout, reference_route, obstacle_clearance,
)
from gym_art.quadrotor_multi.obstacles.priority_layouts import priority_catalogue
from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue


def spec(layout, n=8, seed=0):
    return LayoutSpec('suite', layout_id=layout, num_agents=n, seed=seed,
                      room_dims=(14., 14., 10.))


@pytest.mark.parametrize('layout', [r['id'] for r in priority_catalogue()])
@pytest.mark.parametrize('n', [1, 4, 8])
def test_priority_geometry_is_feasible(layout, n):
    plan = build_layout(spec(layout, n))
    assert reference_route(plan).shape[0] > 1
    assert np.all(plan.starts[:, 1] < 0.)
    np.testing.assert_array_equal(plan.starts[:, 2], 2.)
    np.testing.assert_array_equal(plan.goals[:, 2], 2.)
    clear = plan.spec.agent_radius + plan.spec.safety_margin
    for positions in (plan.starts, plan.goals):
        assert obstacle_clearance(positions[:, :2], plan.obstacles, .3).min() >= clear - 1e-8


@pytest.mark.parametrize('family', ['shallow_pocket', 'entry_corridor'])
@pytest.mark.parametrize('seed', [0, 1, 7])
def test_priority_controls_change_only_the_cap(family, seed):
    opened = build_layout(spec(family + '_open', seed=seed))
    closed = build_layout(spec(family + '_deadend', seed=seed))
    np.testing.assert_array_equal(opened.starts, closed.starts)
    np.testing.assert_array_equal(opened.goals, closed.goals)
    a = {tuple(p) for p in opened.obstacles}
    b = {tuple(p) for p in closed.obstacles}
    assert a < b


def test_shallow_pocket_has_external_goals_and_small_nominal_capacity():
    plan = build_layout(spec('shallow_pocket_deadend'))
    assert plan.metadata['nominal_planar_slots'] == 2
    assert plan.metadata['nominal_planar_slots'] < plan.spec.num_agents
    assert np.all(plan.goals[:, 1] > plan.metadata['depth'] + .3)
    assert plan.metadata['vertical_motion_restricted'] is False


def test_entry_corridor_has_eight_distinct_internal_goals():
    plan = build_layout(spec('entry_corridor_deadend'))
    assert np.all(plan.goals[:, 1] > 0.)
    assert np.all(plan.goals[:, 1] < plan.metadata['depth'] - .3)
    assert np.all(plan.goals[:, 0] == 0.)
    assert np.diff(np.sort(plan.goals[:, 1])).min() >= .35 - 1e-8


@pytest.mark.parametrize('layout', [r['id'] for r in priority_catalogue()])
def test_small_priority_scenes_reject_an_infeasible_swarm_size(layout):
    with pytest.raises(ValueError, match='1..8'):
        build_layout(spec(layout, n=100))


@pytest.mark.parametrize('angle', [90, 180, 270])
def test_priority_layout_can_be_rotated(angle):
    plan = build_layout(replace(spec('entry_corridor_deadend'), rotation=angle))
    assert reference_route(plan).shape[0] > 1


def test_old_default_suite_is_unchanged():
    assert len(catalogue()) == 18
    assert len(catalogue(include_priority=True)) == 22


@pytest.mark.parametrize('layout', [r['id'] for r in priority_catalogue()])
@pytest.mark.parametrize('mode', ['off', 'mask', 'soft'])
def test_priority_scene_runs_in_the_real_environment(layout, mode):
    import json
    from pathlib import Path
    import gymnasium as gym
    import torch
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
    from swarm_rl.evaluate_layouts import load_checkpoint
    from swarm_rl.run_layout_suite import make_suite_config
    from swarm_rl.train import register_swarm_components

    original = json.loads(Path('experiments/swarm100-1b/config.json').read_text())
    cfg = make_suite_config(original, layout, 8, .05, numba=False)
    cfg.quads_static_priority = mode
    cfg.layout_seed = 3
    np.random.seed(3)
    register_swarm_components()
    env = make_quadrotor_env_multi(cfg)

    try:
        obs, _ = env.reset()
        raw = env.unwrapped
        assert raw.static_priority == mode
        assert np.asarray(obs).shape == (8, 40)
        np.testing.assert_allclose(raw.pos, raw.layout_plan.starts)
        actions = np.zeros((8, 4))
        checkpoint = os.environ.get('QUAD_PRETRAINED_CHECKPOINT')

        if checkpoint:
            torch.set_num_threads(1)
            model = create_actor_critic(
                cfg,
                gym.spaces.Dict({'obs': env.observation_space}),
                env.action_space,
            )
            model.load_state_dict(load_checkpoint(Path(checkpoint))['model'], strict=True)
            model.eval()

            with torch.inference_mode():
                model(
                    {'obs': torch.as_tensor(np.asarray(obs), dtype=torch.float32)},
                    torch.zeros(8, get_rnn_size(cfg)),
                )
                actions = model.action_distribution().means.cpu().numpy()

            assert np.isfinite(actions).all()

        obs, *_ = env.step(np.clip(actions, -1., 1.))
        assert np.asarray(obs).shape == (8, 40)
        assert np.isfinite(obs).all()
    finally:
        env.close()
