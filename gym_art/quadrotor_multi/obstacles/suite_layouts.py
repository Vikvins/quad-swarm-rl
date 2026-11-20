"""Paired tunnel controls and dead ends, using only existing cylinder physics.

Tunnels are 2.5D passages: walls span room height. No new roof sensor or
3D collision primitive is introduced into a policy trained with planar SDF.
"""
from dataclasses import replace
import numpy as np

from gym_art.quadrotor_multi.obstacles.layouts import (
    LayoutPlan, _wall, _zone, validate_layout,
)

from gym_art.quadrotor_multi.obstacles.priority_layouts import (
    priority_catalogue, build_priority_layout,
)


FAMILIES = {
    'straight': ('Прямой проход', 1.6, [(0., -5.), (0., 3.)], (0., -3.7), (0., 4.8), 0.),
    'narrow': ('Узкий проход', .9, [(0., -5.), (0., 3.)], (0., -3.7), (0., 4.8), 0.),
    'elbow': ('Г-образный проход', 1.6, [(-3., -5.), (-3., 0.), (3., 0.), (3., 3.)], (-3., -3.7), (3., 4.8), 0.),
    'zigzag': ('S-образный проход', 1.4, [(-2., -5.), (-2., -2.), (2., -2.), (2., 1.2), (-2., 1.2), (-2., 3.8)], (-2., -3.8), (-2., 5.5), 0.),
    'funnel': ('Сужающийся проход', 1.2, None, (0., -3.7), (0., 4.8), 0.),
    'wide_u': ('Широкая U-камера', 3.2, [(0., -2.), (0., 2.)], (0., .1), (0., 4.), 0.),
    'fork': ('Т-развилка', 1.6, None, (0., -3.7), (5.5, 0.), 90.),
    'comb': ('Проход с боковыми ответвлениями', 1.4, None, (0., -3.7), (0., 4.8), 0.),
}


def catalogue(include_priority=False):
    rows = [dict(id='random_baseline', family='random', title='Исходная случайная среда', has_dead_end=False, paired_with=None),
            dict(id='open_room', family='open', title='Открытая комната', has_dead_end=False, paired_with=None)]
    for family, (title, width, *_rest) in FAMILIES.items():
        for capped in (False, True):
            suffix = 'deadend' if capped else 'open'
            other = 'open' if capped else 'deadend'
            rows.append(dict(id=f'{family}_{suffix}', family=family,
                             title=title + (' с тупиком' if capped else ' без тупика'),
                             width=width, has_dead_end=capped, paired_with=f'{family}_{other}'))
    if include_priority:
        rows.extend(priority_catalogue())
    return rows


def _rectangles(path, a):
    rectangles = []
    for p, q in zip(path, path[1:]):
        if p[0] == q[0]:
            rectangles.append((p[0]-a, p[0]+a, min(p[1],q[1]), max(p[1],q[1])))
        elif p[1] == q[1]:
            rectangles.append((min(p[0],q[0]), max(p[0],q[0]), p[1]-a, p[1]+a))
        else:
            raise ValueError('Only axis-aligned centreline segments are supported')
    return rectangles


def _union_boundary(rectangles, portals):
    """Exact rectangle-union boundary, with selected end portals removed."""
    xs = sorted({r[i] for r in rectangles for i in (0, 1)})
    ys = sorted({r[i] for r in rectangles for i in (2, 3)})
    inside = np.zeros((len(xs)-1, len(ys)-1), bool)
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            x, y = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2
            inside[i,j] = any(x0 < x < x1 and y0 < y < y1 for x0,x1,y0,y1 in rectangles)
    segments = []
    for i,j in np.argwhere(inside):
        candidates = []
        if i == 0 or not inside[i-1,j]: candidates.append(((xs[i],ys[j]),(xs[i],ys[j+1])))
        if i == len(xs)-2 or not inside[i+1,j]: candidates.append(((xs[i+1],ys[j]),(xs[i+1],ys[j+1])))
        if j == 0 or not inside[i,j-1]: candidates.append(((xs[i],ys[j]),(xs[i+1],ys[j])))
        if j == len(ys)-2 or not inside[i,j+1]: candidates.append(((xs[i],ys[j+1]),(xs[i+1],ys[j+1])))
        for p,q in candidates:
            mid = (np.asarray(p)+q)/2
            if any(np.isclose(mid[axis],value) and low-1e-8 <= mid[1-axis] <= high+1e-8
                   for axis,value,low,high in portals):
                continue
            segments.append((p,q))
    return segments


def _portal(point, direction, a):
    axis = 0 if abs(direction[0]) > abs(direction[1]) else 1
    return (axis, point[axis], point[1-axis]-a, point[1-axis]+a)


def build_suite_layout(spec):
    layout_id = spec.layout_id
    if layout_id in {r['id'] for r in priority_catalogue()}:
        return build_priority_layout(spec)
    entry = next((r for r in catalogue() if r['id'] == layout_id), None)
    if entry is None or layout_id == 'random_baseline':
        raise ValueError(f'Unknown structured suite layout: {layout_id}')
    rng = np.random.default_rng(spec.seed)
    if layout_id == 'open_room':
        spec = replace(spec, width=4.)
        segments, start, goal, goal_angle = [], (0., -3.7), (0., 4.8), 0.
    else:
        family = entry['family']
        title,width,path,start,goal,goal_angle = FAMILIES[family]
        spec = replace(spec, width=width)
        a = width/2 + spec.obstacle_size/2
        capped = entry['has_dead_end']
        if family == 'funnel':
            segments = [((-2.3,-5.),(-a,-1.)),((-a,-1.),(-a,3.)),
                        ((2.3,-5.),(a,-1.)),((a,-1.),(a,3.))]
            if capped: segments.append(((-a,3.),(a,3.)))
        elif family == 'fork':
            rectangles = [( -a,a,-5.,0.),(-4.,4.,-a,a)]
            portals = [(1,-5.,-a,a),(0,-4.,-a,a)]
            if not capped: portals.append((0,4.,-a,a))
            segments = _union_boundary(rectangles, portals)
        elif family == 'comb':
            rectangles = [(-a,a,-5.,3.),(-4.,4.,-1.5-a,-1.5+a),(-4.,4.,1.-a,1.+a)]
            portals = [(1,-5.,-a,a)]
            if not capped:
                portals += [(1,3.,-a,a)] + [(0,x,y-a,y+a) for x in (-4.,4.) for y in (-1.5,1.)]
            segments = _union_boundary(rectangles, portals)
        else:
            first, second = np.asarray(path[0]), np.asarray(path[1])
            last, prev = np.asarray(path[-1]), np.asarray(path[-2])
            portals = [_portal(first, second-first, a)]
            if not capped: portals.append(_portal(last,last-prev,a))
            segments = _union_boundary(_rectangles(path,a),portals)
    pad = spec.agent_radius + spec.safety_margin + spec.grid_resolution
    # Enough vertical capacity for 100 drones even in the 0.9 m control.
    spec = replace(spec, z_min=1., z_max=4.)
    half = spec.width/2-pad
    def zone(center, angle):
        points = _zone((-half,half),(-.9,.9),spec,rng)
        rad = np.deg2rad(angle)
        rotation = np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
        points[:,:2] = points[:,:2] @ rotation.T + center
        return points
    starts,goals = zone(start,0.),zone(goal,goal_angle)
    xy = np.concatenate([_wall(p,q,spec.wall_spacing) for p,q in segments]) if segments else np.empty((0,2))
    xy = np.unique(np.round(xy,10),axis=0)
    angle = np.deg2rad(spec.rotation)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    xy = xy @ rotation.T
    starts[:,:2] = starts[:,:2] @ rotation.T
    goals[:,:2] = goals[:,:2] @ rotation.T
    segments = [(np.asarray([p,q]) @ rotation.T).tolist() for p,q in segments]
    obstacles = np.column_stack((xy,np.full(len(xy),spec.room_dims[2]/2)))
    metadata = dict(entry, geometry='2.5D full-height cylinder walls',
                    start_center=list(start),goal_center=list(goal),
                    paired_starts_goals=True, portal_defined=False)
    plan = LayoutPlan(spec,obstacles,starts,goals,segments,metadata)
    validate_layout(plan)
    return plan
