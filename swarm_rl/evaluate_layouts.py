"""Frozen-checkpoint evaluation on validated corridor/dead-end layouts, CPU only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import runpy
import shlex
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.layout_metrics import LayoutMetrics
from swarm_rl.preview_layouts import plot_layout
from swarm_rl.train import parse_swarm_cfg, register_swarm_components


def load_checkpoint(path):
    if path.stat().st_size < 1024:
        raise ValueError(f'{path} is too small: download Git LFS content, not a pointer')
    allowed = [np.core.multiarray.scalar, np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        result = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' not in result:
        raise ValueError('Expected a Sample Factory checkpoint with a model state_dict')
    return result


def make_config(args, state, kind, width):
    project = Path(__file__).resolve().parents[1]
    baseline = runpy.run_path(str(project / 'swarm_rl/runs/obstacles/quad_obstacle_baseline.py'))['QUAD_BASELINE_CLI_8']
    cfg = parse_swarm_cfg(shlex.split(baseline)[3:] + ['--device=cpu', '--quads_encoder_type=attention'])
    if args.config:
        for key, value in json.loads(args.config.read_text()).items():
            setattr(cfg, key, value)
    elif not args.allow_inferred_config:
        raise ValueError('Provide --config ORIGINAL_CONFIG.json or explicitly accept --allow_inferred_config')
    else:
        # Reconstruct only the supported architecture, not undocumented training
        # conditions. All evaluation conditions are saved in effective_config.json.
        expected = [('self', 19), ('neighbor', 12), ('obstacle', 9)]
        hidden = state['actor_encoder.self_embed_layer.0.weight'].shape[0]
        for name, size in expected:
            if tuple(state[f'actor_encoder.{name}_embed_layer.0.weight'].shape) != (hidden, size):
                raise ValueError('Unsupported checkpoint architecture; original config is required')
        if state['actor_encoder.attention_layer.w_qs.weight'].shape[0] != 4 * hidden:
            raise ValueError('Expected the supported four-head attention architecture')
        cfg.rnn_size = hidden
        cfg.quads_neighbor_hidden_size = hidden
        cfg.actor_critic_share_weights = False
        cfg.use_rnn = False
        cfg.adaptive_stddev = False
        cfg.normalize_input = False
        cfg.normalize_returns = False
        cfg.quads_obs_repr = 'xyz_vxyz_R_omega_floor'
        cfg.quads_neighbor_obs_type = 'pos_vel'
        cfg.quads_neighbor_visible_num = 2
        cfg.quads_obstacle_obs_type = 'octomap'
        cfg.quads_encoder_type = 'attention'
    cfg.device = 'cpu'
    cfg.quads_num_agents = args.agents
    cfg.quads_room_dims = args.room_dims
    cfg.quads_episode_duration = args.duration
    cfg.quads_obst_size = args.obstacle_size
    cfg.quads_use_obstacles = True
    cfg.quads_use_numba = args.numba
    cfg.quads_render = False
    cfg.quads_sim2real = False
    cfg.visualize_v_value = False
    cfg.replay_buffer_sample_prob = 0.
    cfg.anneal_collision_steps = 0
    cfg.obstacle_layout = kind
    cfg.layout_width = width
    cfg.layout_seed = args.seed
    cfg.layout_rotation = args.rotation
    cfg.corridor_length = args.corridor_length
    cfg.dead_end_depth = args.dead_end_depth
    cfg.wall_spacing = args.wall_spacing
    cfg.with_wandb = False
    return cfg


def _write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_case(args, checkpoint_path, checkpoint, kind, width, output):
    cfg = make_config(args, checkpoint['model'], kind, width)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'effective_config.json').write_text(json.dumps(vars(cfg), indent=2))
    env = make_quadrotor_env_multi(cfg)
    raw = env.unwrapped
    model = create_actor_critic(cfg, gym.spaces.Dict({'obs': env.observation_space}), env.action_space)
    # Never ignore missing/unexpected keys or change the model for a checkpoint.
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    agent_rows, episode_rows = [], []
    inference_times = []
    try:
        with torch.inference_mode():
            for episode in range(args.episodes):
                seed = args.seed + episode
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                raw.layout_episode_index = episode
                obs, _ = env.reset()
                plan = raw.layout_plan
                # Verify real starts, not just the planned coordinates.
                np.testing.assert_allclose(raw.pos, plan.starts, atol=1e-9)
                tracker = LayoutMetrics(plan, raw.control_dt, raw.collision_threshold,
                                        goal_radius=args.goal_radius, goal_hold=args.goal_hold)
                states = torch.zeros(args.agents, get_rnn_size(cfg))
                start = time.perf_counter()
                for tick in range(int(np.ceil(args.duration / raw.control_dt)) + 2):
                    inputs = {'obs': torch.as_tensor(np.asarray(obs), dtype=torch.float32)}
                    began = time.perf_counter()
                    outputs = model(model.normalize_obs(inputs), states)
                    selected = outputs['actions'] if args.stochastic else argmax_actions(model.action_distribution())
                    actions = selected.cpu().numpy()
                    if tick >= 10:
                        inference_times.append(time.perf_counter() - began)
                    if not np.isfinite(actions).all():
                        raise ValueError('Policy produced nonfinite actions')
                    actions = np.clip(actions, env.action_space.low, env.action_space.high)
                    states = outputs['new_rnn_states']
                    obs, _, terminated, truncated, _ = env.step(actions)
                    done = bool(np.any(np.logical_or(terminated, truncated)))
                    terminal = raw.layout_terminal_state
                    positions = terminal['positions'] if done else raw.pos
                    rooms = terminal['room_contacts'] if done else [
                        bool(e.dynamics.crashed_floor or e.dynamics.crashed_wall or e.dynamics.crashed_ceiling)
                        for e in raw.envs]
                    tracker.update(positions, rooms)
                    if done:
                        break
                else:
                    raise RuntimeError('Episode did not terminate at its configured horizon')
                prefix = dict(checkpoint=checkpoint_path.name, env_steps=int(checkpoint.get('env_steps', 0)),
                              layout=kind, width=width, seed=seed, agents=args.agents)
                episode_rows.append(dict(**prefix, **tracker.summary(), wall_seconds=time.perf_counter() - start))
                agent_rows.extend(dict(**prefix, **row) for row in tracker.agent_rows())
                (output / f'layout_seed{seed}.json').write_text(json.dumps(plan.to_dict(), indent=2))
                if episode == 0 or args.save_trajectories:
                    trajectories = np.asarray(tracker.positions, dtype=np.float32)
                    plot_layout(plan, output / f'trajectory_seed{seed}.png', trajectories)
                    np.savez_compressed(output / f'trajectory_seed{seed}.npz', positions=trajectories)
                # Flush per episode so an interrupted benchmark retains finished cases.
                _write_csv(output / 'episodes.csv', episode_rows)
                _write_csv(output / 'agents.csv', agent_rows)
                log.info('%s width=%s seed=%s success=%.3f collisions=%.3f',
                         kind, width, seed, episode_rows[-1]['success_rate'], episode_rows[-1]['collision_rate'])
    finally:
        env.close()
    summary = dict(checkpoint=checkpoint_path.name, env_steps=int(checkpoint.get('env_steps', 0)),
                   layout=kind, width=width, agents=args.agents, episodes=len(episode_rows),
                   policy_mode='stochastic' if args.stochastic else 'deterministic')
    for metric in ['success_rate', 'collision_rate', 'obstacle_collision_rate', 'neighbor_collision_rate',
                   'room_collision_rate', 'crossed_portal_rate', 'timeout_rate', 'stalled_rate',
                   'mean_path_length', 'mean_final_distance']:
        values = np.asarray([r[metric] for r in episode_rows], dtype=float)
        summary[metric] = float(values.mean())
        # Resample whole episodes, not correlated drones within an episode.
        if len(values) >= 2:
            rng = np.random.default_rng(927)
            means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            summary[metric + '_ci_low'], summary[metric + '_ci_high'] = map(float, np.quantile(means, [.025, .975]))
        else:
            summary[metric + '_ci_low'] = summary[metric + '_ci_high'] = None
    summary['mean_inference_ms_per_agent'] = float(np.mean(inference_times) * 1000 / args.agents) if inference_times else None
    summary['inference_timing_note'] = 'CPU actor+critic batch latency divided by agents; not isolated single-agent latency; first ten ticks excluded'
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--allow_inferred_config', action='store_true')
    parser.add_argument('--layouts', nargs='+', choices=['corridor', 'dead_end'], default=['corridor', 'dead_end'])
    parser.add_argument('--widths', nargs='+', type=float, default=[1.2])
    parser.add_argument('--agents', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--duration', type=float, default=15.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument('--room_dims', nargs=3, type=float, default=[10., 10., 10.])
    parser.add_argument('--obstacle_size', type=float, default=.6)
    parser.add_argument('--corridor_length', type=float, default=5.)
    parser.add_argument('--dead_end_depth', type=float, default=3.)
    parser.add_argument('--wall_spacing', type=float, default=.45)
    parser.add_argument('--goal_radius', type=float, default=.3)
    parser.add_argument('--goal_hold', type=float, default=.5)
    parser.add_argument('--numba', action='store_true')
    parser.add_argument('--stochastic', action='store_true', help='Sample the policy instead of taking distribution means')
    parser.add_argument('--save_trajectories', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.agents < 1 or args.episodes < 1 or args.duration <= 0 or args.goal_radius <= 0 or args.goal_hold <= 0:
        parser.error('Agents, episodes, duration, goal radius and hold time must be positive')
    if not args.config and not args.allow_inferred_config:
        parser.error('Missing original config: pass --config, or explicitly --allow_inferred_config')
    torch.set_num_threads(1)
    register_swarm_components()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    root = args.output or Path('results') / f'layout_eval_{stamp}'
    root.mkdir(parents=True, exist_ok=False)
    git_hash = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    project = Path(__file__).resolve().parents[1]
    code_paths = subprocess.check_output(['git', '-C', str(project), 'ls-files', '--cached', '--others',
                                          '--exclude-standard', '--', '*.py'], text=True).splitlines()
    manifest = dict(code_commit=git_hash, config_source=str(args.config) if args.config else 'architecture inferred; declared evaluation defaults',
                    torch=torch.__version__, numpy=np.__version__, arguments={k: str(v) for k, v in vars(args).items()},
                    source_sha256={p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in code_paths}, checkpoints=[])
    summaries = []
    for index, path in enumerate(args.checkpoint):
        checkpoint = load_checkpoint(path)
        manifest['checkpoints'].append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            env_steps=int(checkpoint.get('env_steps', 0))))
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        for kind in args.layouts:
            for width in args.widths:
                output = root / f'ckpt{index}_{path.stem}' / f'{kind}_w{width:g}'
                summaries.append(evaluate_case(args, path, checkpoint, kind, width, output))
                _write_csv(root / 'summary.csv', summaries)
    log.info('Results saved to %s', root.resolve())
    return root


if __name__ == '__main__':
    main()
