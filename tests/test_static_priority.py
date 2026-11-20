import json
import sys

import numpy as np
import pytest

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.train import parse_swarm_cfg


def config(kind, agents, priority=None, scale=None):
    args = [
        '--env=quadrotor_multi', '--device=cpu', '--seed=123',
        f'--quads_num_agents={agents}', '--quads_neighbor_visible_num=2',
        '--quads_neighbor_obs_type=pos_vel', '--quads_obs_repr=xyz_vxyz_R_omega_floor',
        '--quads_obstacle_obs_type=octomap', '--quads_use_obstacles=True',
        '--quads_obst_size=0.6', '--quads_encoder_type=attention',
        '--quads_mode=mix', f'--obstacle_layout={kind}',
        '--quads_use_numba=False', '--quads_episode_duration=2',
        '--replay_buffer_sample_prob=0', '--anneal_collision_steps=0',
    ]
    if priority is not None:
        args.append(f'--quads_static_priority={priority}')
    if scale is not None:
        args.append(f'--quads_static_priority_scale={scale}')
    cfg = parse_swarm_cfg(args)
    cfg.quads_render = False
    return cfg


def swarm_state(kind, agents, steps=15):
    """An environment after a few random steps, plus the own (self) part of every drone's observation."""
    env = make_quadrotor_env_multi(config(kind, agents))
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(steps):
        env.step(rng.uniform(-1, 1, (agents, 4)))
    raw = env.unwrapped
    self_obs = np.array([e.state_vector(e) for e in raw.envs])
    return env, raw, self_obs


def neighbor_obs(raw, self_obs, mode, scale=.7):
    raw.static_priority, raw.static_priority_scale = mode, scale
    return raw.add_neighborhood_obs(self_obs.copy())


@pytest.mark.parametrize('kind', ['random', 'corridor', 'dead_end'])
def test_flag_reaches_environment_and_default_is_off(kind):
    for priority, expected in [(None, 'off'), ('off', 'off'), ('mask', 'mask'), ('soft', 'soft')]:
        env = make_quadrotor_env_multi(config(kind, 8, priority, scale=.5))
        try:
            assert env.unwrapped.static_priority == expected
            assert env.unwrapped.static_priority_scale == .5
            obs, _ = env.reset()
            assert np.asarray(obs).shape == (8, 40)
            obs, *_ = env.step(np.zeros((8, 4)))
            assert np.isfinite(obs).all()
        finally:
            env.close()


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
def test_off_is_identical_to_original_observation(kind):
    env, raw, self_obs = swarm_state(kind, 8)
    try:
        original = raw.extend_obs_space(self_obs.copy(), closest_drones=raw.neighborhood_indices())
        np.testing.assert_array_equal(neighbor_obs(raw, self_obs, 'off'), original)
    finally:
        env.close()


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
def test_mask_shows_only_senior_neighbors(kind):
    env, raw, self_obs = swarm_state(kind, 8)
    try:
        obs = neighbor_obs(raw, self_obs, 'mask')
        far = raw.clip_neighbor_space_max_box.reshape(2, 6)[:, :3]
        # drone 0 has no seniors: both slots are 'nobody around'
        np.testing.assert_allclose(obs[0, 19:31].reshape(2, 6)[:, :3], far)
        np.testing.assert_array_equal(obs[0, 19:31].reshape(2, 6)[:, 3:], 0.)
        # drone 1 has one senior: slot 1 is drone 0, slot 2 is empty
        rel0 = np.clip(raw.pos[0] - raw.pos[1], -far[0], far[0])
        np.testing.assert_allclose(obs[1, 19:22], rel0, atol=1e-6)
        np.testing.assert_allclose(obs[1, 25:28], far[1])
        # every other drone sees two seniors only
        indices, missing = raw.neighborhood_indices_seniors()
        for i in range(2, 8):
            assert missing[i] == 0 and all(j < i for j in indices[i])
        # the own part and the obstacle part are untouched
        np.testing.assert_array_equal(obs[:, :19], self_obs)
    finally:
        env.close()


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
def test_soft_scales_only_senior_positions(kind):
    env, raw, self_obs = swarm_state(kind, 8)
    try:
        off = neighbor_obs(raw, self_obs, 'off')
        soft = neighbor_obs(raw, self_obs, 'soft', scale=.5)
        indices = raw.neighborhood_indices()
        for i in range(8):
            for slot, j in enumerate(indices[i]):
                a = 19 + 6 * slot
                factor = .5 if j < i else 1.
                np.testing.assert_allclose(soft[i, a:a + 3], off[i, a:a + 3] * factor, atol=1e-6)
                np.testing.assert_array_equal(soft[i, a + 3:a + 6], off[i, a + 3:a + 6])
    finally:
        env.close()


def test_suite_runner_refuses_to_mix_priority_modes_in_one_folder(tmp_path, monkeypatch):
    from swarm_rl import run_layout_suite
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(static_priority='off', static_priority_scale=.7)))
    monkeypatch.setattr(sys, 'argv', ['run_layout_suite', '--checkpoint', 'x.pth', '--config', 'x.json',
                                      '--output', str(tmp_path), '--static_priority', 'mask'])
    with pytest.raises(SystemExit, match='already holds results for static_priority=off'):
        run_layout_suite.main()


def test_comparison_pairs_episodes_and_reports_differences(tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('compare', Path(__file__).resolve().parents[1] / 'scripts/compare_static_priority.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    rows = dict(off=[(0, 1.0, 0.0), (1, 0.5, 0.5), (2, 0.0, 1.0)], mask=[(0, 0.5, 0.5), (1, 1.0, 0.0)])
    for name, episodes in rows.items():
        (tmp_path / name).mkdir()
        lines = ['agents,layout_id,seed,success_rate,collision_rate'] + \
                [f'8,straight_open,{seed},{s},{c}' for seed, s, c in episodes]
        (tmp_path / name / 'episodes.csv').write_text('\n'.join(lines))
    out = tmp_path / 'cmp.csv'
    monkeypatch.setattr(sys, 'argv', ['compare', '--runs', f'off={tmp_path / "off"}', f'mask={tmp_path / "mask"}',
                                      '--output', str(out)])
    compare.main()
    import pandas as pd
    table = pd.read_csv(out, header=[0, 1], index_col=[0, 1])
    # seed 2 exists only in the off run and is dropped: means over seeds 0 and 1
    assert table[('success_rate', 'off')].iloc[0] == pytest.approx(.75)
    assert table[('success_rate', 'mask')].iloc[0] == pytest.approx(.75)
    assert table[('success_rate', 'mask-off')].iloc[0] == pytest.approx(0.)
    assert table[('collision_rate', 'mask-off')].iloc[0] == pytest.approx(0.)
