"""
Inference-only evaluation of a trained policy with and without external static priority.

No training and no changes to the network. The modes are compared on the same starts and goals:

  off       the policy as it was trained: every drone sees its K closest neighbors.
  mask      external static priority by drone number: drone 0 is the most senior, drone N-1 the most junior.
            A drone only sees neighbors that are senior to it (K closest among drones with a smaller number),
            so seniors fly as if juniors did not exist and juniors avoid everyone.
            When a drone has fewer than K seniors (drone 0 has none), the empty slot shows
            a neighbor far away with zero relative velocity.
  soft      soft static priority by drone number: every drone sees its K closest neighbors as usual,
            but a senior neighbor is shown closer than it really is (relative position times --soft_scale),
            so juniors start avoiding earlier and keep a larger distance, while seniors still avoid juniors.

  dynamic   temporary hover goals based on a local inference-time priority.

For a model fine-tuned with --quads_neighbor_obs_type=pos_vel_prio the priority is not imposed from outside:
the environment puts it into the observation, and the modes mean
  off       all priorities equal (the priority feature is 0 for every neighbor)
  mask      priority by drone number (variant 2)
  random    a random order in every episode

Every episode is run once in each mode with the same random seed, so both modes get the same
spawn positions, goals and obstacles.

Metrics per episode (the same definitions as the environment uses for training logs):
  success_rate     share of drones that reached the goal and never collided with a drone or an obstacle
  reached_rate     share of drones that reached the goal (collisions allowed)
  col_rate         share of drones with at least one collision (drone or obstacle)
  drone_col_rate   share of drones with at least one drone-drone collision
  deadlock_rate    share of drones without collisions that never reached the goal
  num_collisions   number of drone-drone collisions after the drones settled
  time_to_goal     mean time (s) until a drone first reached the goal, over drones that reached it

Example (from the repository root):
  python -m swarm_rl.eval_static_priority \
      --cfg path/to/config.json \
      --checkpoint path/to/checkpoint_000781254_1000005120.pth \
      --num_agents 8 100 --episodes 20 --out results
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import types

import gymnasium as gym
import numpy as np
import torch

try:
    import bezier  # noqa: F401  (only needed by the bezier scenarios)
except ImportError:
    sys.modules['bezier'] = types.ModuleType('bezier')

from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from swarm_rl.dynamic_priority import DynamicPriority, add_dynamic_priority_args, dynamic_priority_config
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.evaluate_layouts import load_checkpoint
from swarm_rl.train import parse_swarm_cfg, register_swarm_components


# ---------------------------------------------------------------- static priority
def neighborhood_indices_static_priority(self):
    """K closest neighbors among senior drones only (smaller number = more senior).
    Uses the same closeness metric as the original neighborhood_indices.
    Missing slots are filled with the drone's own index and replaced by a far neighbor afterwards."""
    k = self.num_use_neighbor_obs
    result, missing = [], np.zeros(self.num_agents, dtype=int)
    for i in range(self.num_agents):
        seniors = np.arange(i)
        if len(seniors) == 0:
            chosen = np.array([], dtype=int)
        else:
            rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=seniors)
            rel_dist = np.maximum(np.linalg.norm(rel_pos, axis=1), 0.01)
            new_rel_dist = rel_dist + np.sum(rel_pos / rel_dist[:, None] * rel_vel, axis=1)
            chosen = seniors[new_rel_dist.argsort()[:k]]
        missing[i] = k - len(chosen)
        result.append(np.concatenate([chosen, np.full(missing[i], i, dtype=int)]))
    self._prio_missing = missing
    return result


def add_neighborhood_obs_static_priority(self, obs):
    indices = neighborhood_indices_static_priority(self)
    obs_ext = self.extend_obs_space(obs, closest_drones=indices)
    self_size = obs_ext.shape[1] - self.clip_neighbor_space_length
    k = self.num_use_neighbor_obs
    per = self.clip_neighbor_space_length // k  # 6 for pos_vel
    for i in np.nonzero(self._prio_missing)[0]:
        for slot in range(k - self._prio_missing[i], k):
            a = self_size + slot * per
            obs_ext[i, a:a + 3] = self.clip_neighbor_space_max_box[slot * per:slot * per + 3]  # far away
            obs_ext[i, a + 3:a + per] = 0.0  # zero relative velocity
    return obs_ext


def add_neighborhood_obs_soft_priority(self, obs):
    indices = self.neighborhood_indices()
    obs_ext = self.extend_obs_space(obs, closest_drones=indices)
    self_size = obs_ext.shape[1] - self.clip_neighbor_space_length
    k = self.num_use_neighbor_obs
    per = self.clip_neighbor_space_length // k
    for i in range(self.num_agents):
        for slot, j in enumerate(indices[i]):
            if j < i:  # the neighbor is senior: show it closer
                a = self_size + slot * per
                obs_ext[i, a:a + 3] *= self._soft_scale
    return obs_ext


def unwrap(env):
    while hasattr(env, 'env'):
        env = env.env
    return env


# ---------------------------------------------------------------- one episode
# old mode names are still accepted
MODE_ALIASES = {'none': 'off', 'fixed_id': 'mask'}


def run_episode(env, core, actor_critic, cfg, seed, mode, deterministic, soft_scale=0.7, dynamic_config=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    for i, e in enumerate(core.envs):  # every drone also has its own generator for its spawn point
        e._seed(seed * 1000 + i)

    mode = MODE_ALIASES.get(mode, mode)
    core.track_priority_neighbors = False
    controller = None
    if mode == 'dynamic':
        if getattr(core, 'neighbor_obs_type', None) == 'pos_vel_prio':
            raise ValueError('Dynamic priority requires a checkpoint with the original pos_vel observations')
        controller = DynamicPriority(core, dynamic_config)
    if getattr(core, 'neighbor_obs_type', None) == 'pos_vel_prio':
        # the model was trained with the priority feature: priority comes from the environment, no masking
        core.add_neighborhood_obs = types.MethodType(type(core).add_neighborhood_obs, core)
        core.priority_mode = {'off': 'none', 'mask': 'fixed_id', 'random': 'random_perm'}[mode]
    elif hasattr(core, 'static_priority'):
        # the environment has the --quads_static_priority switch: use it
        core.static_priority = 'off' if controller else mode
        core.static_priority_scale = soft_scale
    elif mode == 'mask':
        core.add_neighborhood_obs = types.MethodType(add_neighborhood_obs_static_priority, core)
    elif mode == 'soft':
        core._soft_scale = soft_scale
        core.add_neighborhood_obs = types.MethodType(add_neighborhood_obs_soft_priority, core)
    else:
        core.add_neighborhood_obs = types.MethodType(type(core).add_neighborhood_obs, core)

    obs, _ = env.reset()
    n = core.num_agents
    rnn_states = torch.zeros(n, get_rnn_size(cfg))
    low, high = env.action_space.low, env.action_space.high
    time_to_goal = np.full(n, np.nan)
    collided = np.zeros(n, dtype=bool)
    step = 0
    while True:
        with torch.no_grad():
            if controller:
                normalized = controller.policy_inputs(actor_critic, obs, rnn_states)
            else:
                normalized = prepare_and_normalize_obs(actor_critic, {'obs': torch.as_tensor(np.array(obs), dtype=torch.float32)})
            out = actor_critic(normalized, rnn_states)
            actions = argmax_actions(actor_critic.action_distribution()) if deterministic else out['actions']
        rnn_states = out['new_rnn_states']
        actions = np.clip(actions.cpu().numpy(), low, high)
        obs, rew, terminated, truncated, infos = env.step(list(actions))
        step += 1
        if np.any(terminated) or np.any(truncated):
            # the environment resets itself on the last step, so its per-drone flags are already cleared here
            break
        reached = np.array(core.reached_goal, dtype=bool)
        newly = reached & np.isnan(time_to_goal)
        time_to_goal[newly] = step * core.control_dt
        collided |= (np.array(core.agent_col_agent) == 0) | (np.array(core.agent_col_obst) == 0)

    stats = infos[0]['episode_extra_stats']
    reached_final = ~np.isnan(time_to_goal)
    episode = dict(
        success_rate=stats['metric/agent_success_rate'],
        reached_rate=float(reached_final.mean()),
        col_rate=stats['metric/agent_col_rate'],
        drone_col_rate=stats['metric/agent_neighbor_col_rate'],
        deadlock_rate=stats['metric/agent_deadlock_rate'],
        num_collisions=stats['num_collisions_after_settle'],
        time_to_goal=float(np.nanmean(time_to_goal)) if reached_final.any() else float('nan'),
    )
    per_drone = [dict(drone=i, reached=int(reached_final[i]), collided=int(collided[i]),
                      time_to_goal=time_to_goal[i]) for i in range(n)]
    return episode, per_drone


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True, help='config.json / cfg.json of the training run')
    p.add_argument('--checkpoint', required=True, help='.pth checkpoint')
    p.add_argument('--num_agents', type=int, nargs='+', default=[8, 100])
    p.add_argument('--modes', nargs='+', default=['off', 'mask'],
                   choices=['off', 'mask', 'soft', 'dynamic', 'random', 'none', 'fixed_id'],
                   help='off: no priority; mask: static priority, seniors do not see juniors; '
                        'soft: static priority, juniors see seniors closer (see --soft_scale). '
                        'dynamic: temporary hover goals. Compare modes on the same episodes, e.g. --modes off mask dynamic')
    p.add_argument('--soft_scale', type=float, default=0.7,
                   help='soft mode: a junior sees a senior neighbor at this share of the real distance')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--scenario', default=None, help='quads_mode, by default the one from the cfg')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stochastic', action='store_true', help='sample actions instead of taking the mean')
    p.add_argument('--out', default='results_static_priority')
    p.add_argument('--label', default='', help='name of the model in the result tables, e.g. base or finetuned')
    add_dynamic_priority_args(p)
    args = p.parse_args()
    try:
        dynamic_config = dynamic_priority_config(args) if 'dynamic' in args.modes else None
    except ValueError as error:
        p.error(str(error))

    register_swarm_components()
    torch.set_num_threads(1)
    with open(args.cfg) as f:
        base = json.load(f)
    base.update(device='cpu', quads_render=False, visualize_v_value=False, replay_buffer_sample_prob=0.0,
                with_wandb=False)
    if args.scenario:
        base['quads_mode'] = args.scenario

    ckpt = load_checkpoint(Path(args.checkpoint))
    os.makedirs(args.out, exist_ok=True)
    ep_rows, drone_rows = [], []

    for n in args.num_agents:
        cfg = parse_swarm_cfg(['--env=quadrotor_multi', '--device=cpu'], evaluation=True)
        vars(cfg).update(base, quads_num_agents=n)
        env = make_quadrotor_env('quadrotor_multi', cfg=cfg)
        core = unwrap(env)
        actor_critic = create_actor_critic(cfg, gym.spaces.Dict({'obs': env.observation_space}), env.action_space)
        actor_critic.load_state_dict(ckpt['model'])
        actor_critic.eval()

        for e in range(args.episodes):
            seed = args.seed + e
            for mode in args.modes:
                t0 = time.time()
                episode, per_drone = run_episode(env, core, actor_critic, cfg, seed, mode, not args.stochastic,
                                                 soft_scale=args.soft_scale, dynamic_config=dynamic_config)
                dt = time.time() - t0
                ep_rows.append(dict(model=args.label, scenario=base['quads_mode'], num_agents=n, mode=MODE_ALIASES.get(mode, mode), episode=e, seed=seed, wall_time_s=round(dt, 1), **episode))
                drone_rows += [dict(model=args.label, scenario=base['quads_mode'], num_agents=n, mode=MODE_ALIASES.get(mode, mode), episode=e, **d) for d in per_drone]
                print(f'N={n:3d} {mode:8s} ep {e:3d}: success {episode["success_rate"]:.2f}  '
                      f'reached {episode["reached_rate"]:.2f}  col {episode["col_rate"]:.2f}  '
                      f'time {episode["time_to_goal"]:.2f}s  ({dt:.0f}s)', flush=True)
        env.close()

    import pandas as pd
    ep_df, drone_df = pd.DataFrame(ep_rows), pd.DataFrame(drone_rows)
    ep_df.to_csv(os.path.join(args.out, 'episodes.csv'), index=False)
    drone_df.to_csv(os.path.join(args.out, 'drones.csv'), index=False)

    metrics = ['success_rate', 'reached_rate', 'col_rate', 'drone_col_rate', 'deadlock_rate',
               'num_collisions', 'time_to_goal']
    summary = ep_df.groupby(['num_agents', 'mode'])[metrics].mean().round(3)
    summary.to_csv(os.path.join(args.out, 'summary.csv'))

    # time to goal and success by drone number: 4 groups from the most senior to the most junior
    drone_df['group'] = drone_df.groupby('num_agents')['drone'].transform(
        lambda d: pd.qcut(d, q=min(4, d.nunique()), labels=False) + 1)
    by_rank = drone_df.groupby(['num_agents', 'mode', 'group']).agg(
        drones=('drone', lambda d: f'{d.min()}-{d.max()}'),
        reached=('reached', 'mean'), collided=('collided', 'mean'),
        time_to_goal=('time_to_goal', 'mean')).round(3)
    by_rank.to_csv(os.path.join(args.out, 'by_rank.csv'))

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    print('\n=== Mean over episodes ===\n', summary)
    print('\n=== By drone number (group 1 = most senior) ===\n', by_rank)
    print(f'\nSaved to {args.out}/: episodes.csv, drones.csv, summary.csv, by_rank.csv')


if __name__ == '__main__':
    main()
