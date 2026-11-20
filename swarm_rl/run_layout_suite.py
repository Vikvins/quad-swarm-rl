"""Reproducible paired frozen-policy benchmark with the team's diagnostic metrics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import gymnasium as gym
import numpy as np
from numba import njit
import torch
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from drone_influence import InfluenceMetrics
from gym_art.quadrotor_multi.obstacles.layouts import LayoutPlan, LayoutSpec
from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue
from swarm_rl.dynamic_priority import DynamicPriority, add_dynamic_priority_args, dynamic_priority_config
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.evaluate_layouts import load_checkpoint, _write_csv
from swarm_rl.layout_metrics import LayoutMetrics
from swarm_rl.preview_layouts import plot_layout
from swarm_rl.train import parse_swarm_cfg, register_swarm_components

_STATE = None
_ORIGINAL_CONFIG = None


@njit
def seed_numba(seed):
    np.random.seed(seed)


def make_suite_config(original, layout_id, agents, duration, numba=True):
    cfg = parse_swarm_cfg(['--env=quadrotor_multi','--device=cpu'], evaluation=True)
    for key,value in original.items():
        setattr(cfg,key,value)
    cfg.device='cpu'
    cfg.env_gpu_observations=False
    cfg.env_gpu_actions=False
    cfg.quads_num_agents=agents
    cfg.quads_episode_duration=duration
    cfg.quads_use_numba=numba
    cfg.quads_render=False
    cfg.visualize_v_value=False
    cfg.quads_sim2real=False
    cfg.replay_buffer_sample_prob=0.
    cfg.anneal_collision_steps=0
    cfg.eval_deterministic=True
    cfg.env_frameskip=cfg.eval_env_frameskip=1
    cfg.obstacle_layout='random' if layout_id == 'random_baseline' else 'suite'
    cfg.quads_mode='o_random'
    cfg.layout_id=layout_id
    cfg.with_wandb=False
    return cfg


def baseline_plan(raw, seed):
    positions=np.asarray(raw.pos).copy()
    goals=np.asarray([e.goal for e in raw.envs]).copy()
    obstacles=np.asarray(raw.obstacles.pos_arr).reshape(-1,3).copy()
    return LayoutPlan(LayoutSpec('random',num_agents=raw.num_agents,room_dims=tuple(raw.room_dims),
                                 obstacle_size=raw.obst_size,agent_radius=raw.quad_arm,seed=seed),
                      obstacles,positions,goals,[],dict(id='random_baseline',title='Исходная случайная среда',
                                                       has_dead_end=False,paired_with=None))


def _finite_mean(array):
    values=np.asarray(array)[np.isfinite(array)]
    return float(values.mean()) if len(values) else None


def run_episode(task):
    (layout_id, agents, seed, duration, output, probes, every, static_priority,
     static_priority_scale, dynamic_config, diagnostics) = task
    folder=Path(output)/f'n{agents}'/layout_id/f'seed{seed:03d}'
    if (folder/'episode.json').exists():
        return json.loads((folder/'episode.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    cfg=make_suite_config(_ORIGINAL_CONFIG,layout_id,agents,duration)
    cfg.layout_seed=seed
    cfg.quads_static_priority='off' if dynamic_config else static_priority
    cfg.quads_static_priority_scale=static_priority_scale
    effective_config = dict(vars(cfg), inference_priority=static_priority)
    if dynamic_config:
        effective_config['dynamic_priority'] = vars(dynamic_config)
    (folder/'effective_config.json').write_text(json.dumps(effective_config,indent=2))
    env=None
    diag=None
    controller=None
    try:
        with redirect_stdout(io.StringIO()):
            env=make_quadrotor_env_multi(cfg)
            raw=env.unwrapped
            raw.capture_terminal_state=True
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            seed_numba(seed)
            for i, drone in enumerate(raw.envs):
                drone._seed(1000 * seed + i)
            if dynamic_config:
                controller = DynamicPriority(raw, dynamic_config)
            obs,_=env.reset()
            plan=baseline_plan(raw,seed) if layout_id=='random_baseline' else raw.layout_plan
            model=create_actor_critic(cfg,gym.spaces.Dict({'obs':env.observation_space}),env.action_space)
            model.load_state_dict(_STATE,strict=True)
            model.eval()
            if diagnostics:
                diag=InfluenceMetrics(folder/'diagnostics',.02,probes,every,None)
        states=torch.zeros(agents,get_rnn_size(cfg))
        tracker=LayoutMetrics(plan,raw.control_dt,raw.collision_threshold)
        # Include reset contacts in the strict metric, without adding a tick.
        from gym_art.quadrotor_multi.obstacles.layouts import obstacle_clearance
        tracker.obstacle_contacts |= obstacle_clearance(plan.starts[:,:2],plan.obstacles,plan.spec.obstacle_size/2)<=plan.spec.agent_radius
        distances=np.linalg.norm(plan.starts[:,None]-plan.starts[None,:],axis=-1)
        np.fill_diagonal(distances,np.inf)
        tracker.neighbor_contacts |= np.any(distances<=raw.collision_threshold,axis=1)
        series=[]
        inference=[]
        reward_total=np.zeros(agents)
        begun=time.perf_counter()
        stats={}
        with torch.inference_mode(), open(os.devnull, 'w') as silent, redirect_stdout(silent):
            for tick in range(int(np.ceil(duration/raw.control_dt))+2):
                t=time.perf_counter()
                if controller:
                    normalized = controller.policy_inputs(model, obs, states)
                else:
                    inputs = {'obs': torch.as_tensor(np.asarray(obs), dtype=torch.float32)}
                    normalized = model.normalize_obs(inputs)
                outputs=model(normalized,states)
                action=model.action_distribution().means.detach().cpu().numpy().copy()
                if tick>=10: inference.append(time.perf_counter()-t)
                if not np.isfinite(action).all(): raise ValueError('Nonfinite actions')
                if diag:
                    diag.observe(cfg,env,model,normalized,states)
                if diag and tick % every == 0:
                    row=dict(tick=tick,time_s=tick*raw.control_dt)
                    row.update({k:_finite_mean(v) for k,v in diag.last_arrays.items()})
                    row.update({k:_finite_mean(v) for k,v in diag.last_influence.items()})
                    row['goal_distance']=float(np.linalg.norm(raw.pos-plan.goals,axis=1).mean())
                    row['physical_speed']=float(np.linalg.norm(raw.vel,axis=1).mean())
                    row['collision_so_far']=float(np.mean(tracker.obstacle_contacts|tracker.neighbor_contacts|tracker.room_contacts))
                    series.append(row)
                states=outputs['new_rnn_states']
                obs,reward,terminated,truncated,infos=env.step(np.clip(action,env.action_space.low,env.action_space.high))
                reward_total+=np.asarray(reward)
                done=bool(np.any(np.logical_or(terminated,truncated)))
                if done:
                    terminal=raw.layout_terminal_state
                    pos=terminal['positions']
                    rooms=terminal['room_contacts']
                    stats=infos[0].get('episode_extra_stats',{})
                else:
                    pos=raw.pos
                    rooms=[bool(e.dynamics.crashed_floor or e.dynamics.crashed_wall or e.dynamics.crashed_ceiling) for e in raw.envs]
                tracker.update(pos,rooms)
                if done: break
            else:
                raise RuntimeError('Configured episode horizon was not reached')
        wall_seconds=time.perf_counter()-begun
        positions=np.asarray(tracker.positions,dtype=np.float32)
        np.savez_compressed(folder/'trajectory.npz',positions=positions)
        layout = json.dumps(plan.to_dict(), indent=2)
        (folder/'layout.json').write_text(layout)
        layout_sha256 = hashlib.sha256(layout.encode()).hexdigest()
        if seed==0:
            if layout_id=='random_baseline':
                plot_baseline(plan,folder/'trajectory.png',positions)
            else:
                plot_layout(plan,folder/'trajectory.png',positions)
        rows=tracker.agent_rows()
        for i,row in enumerate(rows): row['episode_reward']=float(reward_total[i])
        _write_csv(folder/'agents.csv',rows)
        if diag:
            _write_csv(folder/'timeseries.csv',series)
        result=dict(layout_id=layout_id,agents=agents,seed=seed,static_priority=static_priority,
                    layout_sha256=layout_sha256,**tracker.summary(),
                    swarm_success=int(all(r['success'] for r in rows)),
                    official_agent_success=float(stats.get('metric/agent_success_rate',np.nan)),
                    official_collision=float(stats.get('metric/agent_col_rate',np.nan)),
                    official_neighbor_collision=float(stats.get('metric/agent_neighbor_col_rate',np.nan)),
                    official_obstacle_collision=float(stats.get('metric/agent_obst_col_rate',np.nan)),
                    mean_reward=float(reward_total.mean()),wall_seconds=wall_seconds,
                    inference_batch_ms=float(np.mean(inference)*1000) if inference else None,
                    control_dt=raw.control_dt,steps=tracker.ticks,probes=probes,metrics_every=every)
        if controller:
            result.update(dynamic_priority_metric=dynamic_config.metric,
                          yield_tick_rate=controller.yield_ticks / (agents * tracker.ticks),
                          inherited_tick_rate=controller.inherited_ticks / (agents * tracker.ticks))
        for key in ([*diag.last_arrays, *diag.last_influence] if diag else []):
            values=[r[key] for r in series if r.get(key) is not None]
            result['mean_'+key]=float(np.mean(values)) if values else None
        if 'mean_command_change_rate' in result:
            # Full-step CSV is the authoritative source for derivatives; series
            # above is intentionally downsampled only for plots.
            diag.file.flush()
            diag.influence_file.flush()
            for filename,keys in [('metrics.csv',list(diag.last_arrays)),
                                  ('influence.csv',['neighbors_deviation','obstacles_deviation','total_deviation','obstacle_share'])]:
                sums={k:np.zeros(agents) for k in keys}
                counts={k:np.zeros(agents,int) for k in keys}
                with (folder/'diagnostics'/filename).open() as f:
                    for record in csv.DictReader(f):
                        agent=int(record['agent'])
                        for key in keys:
                            value=float(record[key])
                            if np.isfinite(value): sums[key][agent]+=value;counts[key][agent]+=1
                for key in keys:
                    result['mean_'+key]=float(sums[key].sum()/counts[key].sum()) if counts[key].sum() else None
                    for agent in range(agents):
                        rows[agent]['mean_'+key]=float(sums[key][agent]/counts[key][agent]) if counts[key][agent] else None
            _write_csv(folder/'agents.csv',rows)
        (folder/'official_stats.json').write_text(json.dumps(stats,indent=2,default=float))
        temporary = folder/'episode.json.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(folder/'episode.json')
        return result
    finally:
        if diag is not None:
            with redirect_stdout(io.StringIO()): diag.close()
        if env is not None: env.close()


def plot_baseline(plan,output,positions=None):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(7,7))
    for x,y,_ in plan.obstacles: ax.add_patch(Circle((x,y),plan.spec.obstacle_size/2,color='#626c76'))
    ax.scatter(*plan.starts[:,:2].T,s=10,c='#147fbc',label='Starts')
    ax.scatter(*plan.goals[:,:2].T,s=15,c='#218447',marker='x',label='Goals')
    if positions is not None:
        for i in range(positions.shape[1]): ax.plot(positions[:,i,0],positions[:,i,1],c='#aa3070',alpha=.15,lw=.7)
    hx,hy=np.asarray(plan.spec.room_dims[:2])/2
    ax.set(xlim=(-hx,hx),ylim=(-hy,hy),aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f'Original random baseline, N={plan.spec.num_agents}')
    ax.legend();fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)


def initialize(checkpoint,config):
    global _STATE,_ORIGINAL_CONFIG
    torch.set_num_threads(1)
    register_swarm_components()
    _STATE=load_checkpoint(Path(checkpoint))['model']
    _ORIGINAL_CONFIG=json.loads(Path(config).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--agents',type=int,nargs='+',default=[8,100])
    parser.add_argument('--episodes',type=int,default=None,
                        help='episodes per swarm size other than 8 (dynamic default: 24, otherwise 8)')
    parser.add_argument('--episodes8',type=int,default=16)
    parser.add_argument('--duration',type=float,default=15.)
    parser.add_argument('--layouts',nargs='+',default=None)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--probes',type=int,default=8)
    parser.add_argument('--every',type=int,default=10)
    parser.add_argument('--static_priority',default='off',choices=['off','mask','soft','dynamic'],
                        help='static priority by drone number at inference: off = none, mask = a drone sees only '
                             'senior neighbors, soft = senior neighbors are shown closer, dynamic = temporary hover goals. '
                             'Use a separate --output per mode')
    parser.add_argument('--static_priority_scale',type=float,default=.7,
                        help='soft mode: a junior sees a senior neighbor at this share of the real distance')
    parser.add_argument('--no_diagnostics', action='store_true', help='skip per-tick influence probes and CSVs')
    add_dynamic_priority_args(parser)
    args=parser.parse_args()
    dynamic = args.static_priority == 'dynamic'
    try:
        dynamic_config = dynamic_priority_config(args) if dynamic else None
    except ValueError as error:
        parser.error(str(error))
    if args.episodes is None:
        args.episodes = 24 if dynamic else 8
    if args.layouts is None:
        args.layouts = ['random_baseline', 'open_room'] if dynamic else [r['id'] for r in catalogue()]
    previous=args.output/'manifest.json'
    old = None
    if previous.exists():
        # finished episodes are reused on restart, so one output folder must hold one priority mode
        old=json.loads(previous.read_text())
        if (old.get('static_priority','off'),float(old.get('static_priority_scale',.7)))!=(args.static_priority,args.static_priority_scale):
            raise SystemExit(f"{args.output} already holds results for static_priority={old.get('static_priority','off')}; "
                             f"use another --output for static_priority={args.static_priority}")
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('METRICS_TB_EVERY','10')
    os.environ.setdefault('METRICS_TB_AGENTS','0')
    project=Path(__file__).resolve().parents[1]
    manifest=dict(code_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
                  metrics_source_commit='280800367d868d888424a1cab573f98a2d1e87d6',
                  checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  original_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  checkpoint_env_steps=1000005120,arguments={k:str(v) for k,v in vars(args).items()},
                  torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),platform=platform.platform(),
                  diagnostics='Formulas from drone_metrics.py and drone_influence.py; CSV all agents/all steps, TensorBoard swarm means every 10 ticks',
                  numba_random_seeded=True,policy='deterministic frozen weights',
                  source_sha256={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for folder in ('swarm_rl','gym_art') for p in (project/folder).rglob('*.py')})
    for name in ('drone_metrics.py','drone_influence.py'):
        manifest['source_sha256'][name]=hashlib.sha256((project/name).read_bytes()).hexdigest()
    manifest.update(static_priority=args.static_priority,static_priority_scale=args.static_priority_scale,
                    dynamic_priority=vars(dynamic_config) if dynamic else None,
                    episode_rng='1000 * seed + drone_id',
                    diagnostics_enabled=not args.no_diagnostics)
    if old is not None:
        for field in ('dynamic_priority', 'episode_rng', 'diagnostics_enabled', 'checkpoint_sha256',
                      'original_config_sha256', 'source_sha256'):
            if old.get(field) != manifest[field]:
                raise SystemExit(f'{args.output} has different {field}; use another --output')
        for field in ('duration', 'probes', 'every'):
            if old['arguments'].get(field) != manifest['arguments'][field]:
                raise SystemExit(f'{args.output} has different {field}; use another --output')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    tasks=[(layout,n,seed,args.duration,str(args.output),args.probes,args.every,
            args.static_priority,args.static_priority_scale,dynamic_config,not args.no_diagnostics)
           for n in args.agents for seed in range(args.episodes8 if n==8 else args.episodes) for layout in args.layouts]
    # Each subprocess seeds both Python/NumPy and Numba RNG. Independent runs
    # can be resumed from completed episode.json files without discarding data.
    completed=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize,initargs=(str(args.checkpoint),str(args.config))) as pool:
        futures={pool.submit(run_episode,task):task for task in tasks}
        for future in as_completed(futures):
            row=future.result()
            completed.append(row)
            _write_csv(args.output/'episodes.csv',sorted(completed,key=lambda r:(r['agents'],r['layout_id'],r['seed'])))
            print(f"Completed {len(completed)}/{len(tasks)} [{args.static_priority}] N={row['agents']} {row['layout_id']} seed={row['seed']} success={row['success_rate']:.3f} collision={row['collision_rate']:.3f} seconds={row['wall_seconds']:.1f}",flush=True)


if __name__=='__main__':
    main()
