import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from gym_art.quadrotor_multi.obstacles.layouts import LayoutSpec, build_layout
from swarm_rl.run_layout_suite import make_suite_config

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.results / "scenes/n100/narrow_open"
read = lambda mode: json.loads((source / f'{mode}.json').read_text())
base = read('baseline')
config = json.loads((Path(__file__).resolve().parents[1] / "swarm100-1b/config.json").read_text())
cfg = make_suite_config(config, 'narrow_open', 100, 15.)
plan = build_layout(LayoutSpec('suite', num_agents=100, room_dims=tuple(cfg.quads_room_dims),
    obstacle_size=cfg.quads_obst_size, agent_radius=base['collision_threshold'] / cfg.quads_collision_hitbox_radius,
    rotation=cfg.layout_rotation, seed=50000, layout_id='narrow_open'))
assert hashlib.sha256(json.dumps(plan.to_dict(), sort_keys=True).encode()).hexdigest() == base['layout_sha256']
fig, (top, side, curve) = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={'width_ratios':[1,1.15,1.8]}, layout='constrained')
fig.suptitle('Узкий коридор · 100 дронов · одинаковые старты', fontsize=16, weight='bold')
for x, y, _ in plan.obstacles:
    top.add_patch(Circle((x,y), plan.spec.obstacle_size/2, color='#d7dde5'))
top.scatter(plan.starts[:,0],plan.starts[:,1],s=24,color='#1864a6',label='Старт')
top.scatter(plan.goals[:,0],plan.goals[:,1],s=24,color='#689b76',label='Цель')
top.set(xlim=(-1.35,1.35), ylim=(-5.5,6), xlabel='Поперёк, м', ylabel='Вдоль, м', title='Вид сверху')
top.set_aspect('equal')
top.legend(loc='center left', frameon=False, bbox_to_anchor=(-.21,.57))
xs=np.unique(plan.starts[:,0])
for x, marker, color in zip(xs, ['o','x'], ['#1864a6','#77abd6']):
    p=plan.starts[plan.starts[:,0] == x]
    side.scatter(p[:,1],p[:,2],s=28 if marker=='o' else 20, marker=marker,color=color, label=f'x = {x:.2f} м')
side.set(xlabel='Вдоль, м',ylabel='Высота, м',title='Стартовая зона сбоку')
side.legend(loc='upper center',frameon=False,bbox_to_anchor=(.5,-.12),ncols=1)
for mode,label,color,style,width in [
    ('baseline','База','#283b50','-',2.1),('peer_on','Канал включён','#d27a27','--',2.1),
    ('static_brake','Статический фильтр','#35856b','-',2.1)
]:
    if not (source / f'{mode}.json').is_file():
        continue
    row=read(mode)
    ticks=np.array(row['first_contact_ticks']['neighbor'])
    times=np.linspace(0,3,301)
    y=((ticks[:,None]>=0) & (ticks[:,None]*row['dt'] <= times[None,:]+1e-9)).sum(0)
    curve.step(times,y,where='post',label=label,color=color,linestyle=style,linewidth=width)
curve.set(xlabel='Время от старта, с',ylabel='Дронов с контактом',title='Первый контакт с другим дроном',xlim=(0,3),ylim=(0,100))
curve.legend(loc='lower right',frameon=False)
for ax in (side,curve):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.18)
fig.savefig(args.output,dpi=170,bbox_inches='tight')
