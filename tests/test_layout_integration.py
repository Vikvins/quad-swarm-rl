import os
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.train import parse_swarm_cfg, register_swarm_components


def config(kind, agents):
    return parse_swarm_cfg([
        '--env=quadrotor_multi', '--device=cpu', '--seed=123',
        f'--quads_num_agents={agents}', '--quads_neighbor_visible_num=2',
        '--quads_neighbor_obs_type=pos_vel', '--quads_obs_repr=xyz_vxyz_R_omega_floor',
        '--quads_obstacle_obs_type=octomap', '--quads_use_obstacles=True',
        '--quads_obst_size=0.6', '--quads_encoder_type=attention',
        '--quads_mode=mix', f'--obstacle_layout={kind}',
        '--quads_use_numba=False', '--quads_episode_duration=0.03',
        '--replay_buffer_sample_prob=0', '--anneal_collision_steps=0',
        '--actor_critic_share_weights=False', '--adaptive_stddev=False', '--use_rnn=False',
        '--normalize_input=False', '--normalize_returns=False',
    ])


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
@pytest.mark.parametrize('agents', [1, 8, 100])
def test_real_reset_step_shapes_and_terminal_snapshot(kind, agents):
    cfg = config(kind, agents)
    cfg.quads_render = False
    env = make_quadrotor_env_multi(cfg)
    try:
        obs, _ = env.reset()
        raw = env.unwrapped
        plan = raw.layout_plan
        assert np.asarray(obs).shape == (agents, 40)
        assert raw.num_obstacles == len(plan.obstacles)
        np.testing.assert_allclose(raw.pos, plan.starts)
        assert all(e.box == 0 for e in raw.envs)
        if agents == 1:
            np.testing.assert_array_equal(np.asarray(obs)[0, 19:31].reshape(2, 6)[:, 3:], 0.)
            assert np.all(np.asarray(obs)[0, 19:31].reshape(2, 6)[:, :3] > 1.)
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(np.zeros((agents, 4)))
            assert np.asarray(obs).shape == (agents, 40)
            assert np.isfinite(obs).all()
            if np.any(np.logical_or(terminated, truncated)):
                assert raw.layout_terminal_state['positions'].shape == (agents, 3)
                assert raw.layout_terminal_state['room_contacts'].shape == (agents,)
                # The current environment has already reset, but the old terminal
                # state is retained for the evaluator and must not be overwritten.
                assert raw.layout_plan.spec.seed == plan.spec.seed + 1
                assert np.any(np.abs(raw.layout_terminal_state['positions'] - plan.starts) > 1e-12)
                np.testing.assert_allclose(raw.pos, raw.layout_plan.starts)
                break
        else:
            pytest.fail('Environment did not finish the short test episode')
    finally:
        env.close()


def test_random_baseline_still_uses_original_generator_and_spawn_jitter():
    cfg = config('random', 8)
    cfg.quads_render = False
    env = make_quadrotor_env_multi(cfg)
    try:
        obs, _ = env.reset()
        assert np.asarray(obs).shape == (8, 40)
        assert env.unwrapped.layout_plan is None
        assert env.unwrapped.obst_map is not None
        assert all(e.box > 0 for e in env.unwrapped.envs)
    finally:
        env.close()


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
def test_actual_pretrained_checkpoint_loads_strictly_and_produces_finite_actions(kind):
    path = os.environ.get('QUAD_PRETRAINED_CHECKPOINT')
    if not path:
        pytest.skip('Set QUAD_PRETRAINED_CHECKPOINT to run the external-weight integration test')
    from swarm_rl.evaluate_layouts import load_checkpoint
    torch.set_num_threads(1)
    register_swarm_components()
    cfg = config(kind, 8)
    cfg.quads_render = False
    env = make_quadrotor_env_multi(cfg)
    try:
        obs, _ = env.reset()
        model = create_actor_critic(cfg, gym.spaces.Dict({'obs': env.observation_space}), env.action_space)
        checkpoint = load_checkpoint(Path(path))
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        before = model.actor_encoder.self_embed_layer[0].weight.detach().clone()
        with torch.inference_mode():
            out = model(model.normalize_obs({'obs': torch.tensor(np.asarray(obs), dtype=torch.float32)}),
                        torch.zeros(8, get_rnn_size(cfg)))
        assert out['actions'].shape == (8, 4)
        assert torch.isfinite(out['actions']).all()
        torch.testing.assert_close(model.actor_encoder.self_embed_layer[0].weight, before)
    finally:
        env.close()


def test_structured_layout_rejects_collision_replay():
    cfg = config('corridor', 8)
    cfg.replay_buffer_sample_prob = .75
    with pytest.raises(ValueError, match='replay'):
        make_quadrotor_env_multi(cfg)


def test_evaluator_requires_original_config_or_explicit_opt_in():
    from swarm_rl.evaluate_layouts import main
    with pytest.raises(SystemExit):
        main(['--checkpoint', 'not-used.pth'])


def test_evaluator_rejects_lfs_pointer(tmp_path):
    from swarm_rl.evaluate_layouts import load_checkpoint
    path = tmp_path / 'pointer.pth'
    path.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='Git LFS'):
        load_checkpoint(path)


def test_full_evaluator_csv_uses_the_true_terminal_positions(tmp_path):
    checkpoint = os.environ.get('QUAD_PRETRAINED_CHECKPOINT')
    if not checkpoint:
        pytest.skip('Set QUAD_PRETRAINED_CHECKPOINT to run the external-weight evaluator test')
    from swarm_rl.evaluate_layouts import main
    root = main(['--checkpoint', checkpoint, '--allow_inferred_config', '--layouts', 'corridor',
                 '--agents', '1', '--episodes', '1', '--duration', '.05', '--stochastic',
                 '--output', str(tmp_path / 'evaluation')])
    case = root / f'ckpt0_{Path(checkpoint).stem}' / 'corridor_w1.2'
    trajectory = np.load(case / 'trajectory_seed0.npz')['positions']
    plan = json.loads((case / 'layout_seed0.json').read_text())
    with (case / 'agents.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    expected = np.linalg.norm(trajectory[-1, 0] - np.asarray(plan['goals'])[0])
    assert float(rows[0]['final_distance']) == pytest.approx(expected, abs=1e-5)
    assert json.loads((root / 'manifest.json').read_text())['source_sha256']
