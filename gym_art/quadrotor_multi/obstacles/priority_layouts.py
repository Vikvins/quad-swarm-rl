"""Small paired scenes for off/mask/soft evaluation with 1..8 drones."""
from dataclasses import replace
import numpy as np

from gym_art.quadrotor_multi.obstacles.layouts import (
    LayoutPlan, _axis, _wall, _zone, validate_layout,
)


def priority_catalogue():
    rows = []
    for family, title, width, depth in [
        ('shallow_pocket', 'Неглубокий тесный карман', .60, .80),
        ('entry_corridor', 'Узкий вход с целями внутри', .55, 3.60),
    ]:
        for closed in (False, True):
            suffix = 'deadend' if closed else 'open'
            other = 'open' if closed else 'deadend'
            rows.append(dict(
                id=f'{family}_{suffix}', family=family,
                title=title + (' с торцевой стеной' if closed else ' без торцевой стены'),
                width=width, depth=depth, has_dead_end=closed,
                paired_with=f'{family}_{other}', max_agents=8,
            ))
    return rows


def build_priority_layout(spec):
    entry = next((r for r in priority_catalogue() if r['id'] == spec.layout_id), None)
    if entry is None:
        raise ValueError(f'Unknown priority scene: {spec.layout_id}')
    if not 1 <= spec.num_agents <= entry['max_agents']:
        raise ValueError('Priority scenes support 1..8 drones; use 1, 4 or 8 for comparison')
    if spec.wall_spacing > spec.obstacle_size:
        raise ValueError('Wall spacing would leave holes')
    if spec.rotation not in (0, 90, 180, 270):
        raise ValueError('Use rotation 0, 90, 180 or 270')

    spec = replace(spec, width=entry['width'], z_min=2., z_max=2.)
    clearance = spec.agent_radius + spec.safety_margin

    if spec.room_dims[2] <= 2. + clearance:
        raise ValueError('Room too low for the fixed 2 m start/goal height')

    rng = np.random.default_rng(spec.seed)
    width, depth = entry['width'], entry['depth']
    radius = spec.obstacle_size / 2
    a = width / 2 + radius

    segments = [
        ((-a, 0.), (-a, depth)),
        ((a, 0.), (a, depth)),
    ]
    if entry['has_dead_end']:
        segments.append(((-a, depth), (a, depth)))

    starts = _zone((-.9, .9), (-1.5, -.6), spec, rng)

    if entry['family'] == 'shallow_pocket':
        # Goals are outside the pocket, beyond its far end.
        goals = _zone((-.9, .9), (depth + .8, depth + 1.7), spec, rng)
    else:
        # Separate final positions inside the corridor.
        spacing = max(.35, spec.spawn_separation)
        last = depth - radius - clearance - spec.grid_resolution
        ys = last - spacing * np.arange(spec.num_agents)

        if ys.min() <= clearance:
            raise ValueError('Too many safe goal slots inside the corridor')

        goals = np.column_stack((
            np.zeros(spec.num_agents),
            ys,
            np.full(spec.num_agents, 2.),
        ))
        # Do not align destination order with drone-ID priority.
        goals = goals[rng.permutation(spec.num_agents)]

    xs = _axis(
        -width / 2 + clearance,
        width / 2 - clearance,
        spec.spawn_separation,
    )
    ys = _axis(
        clearance,
        depth - radius - clearance,
        spec.spawn_separation,
    )
    nominal_slots = len(xs) * len(ys)

    xy = np.concatenate([
        _wall(p, q, spec.wall_spacing) for p, q in segments
    ])
    xy = np.unique(np.round(xy, 10), axis=0)

    angle = np.deg2rad(spec.rotation)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    xy = xy @ rotation.T
    starts[:, :2] = starts[:, :2] @ rotation.T
    goals[:, :2] = goals[:, :2] @ rotation.T
    segments = [
        (np.asarray([p, q]) @ rotation.T).tolist()
        for p, q in segments
    ]
    obstacles = np.column_stack((
        xy,
        np.full(len(xy), spec.room_dims[2] / 2),
    ))

    metadata = dict(entry)
    metadata.update(
        geometry='2.5D full-height cylinder walls',
        paired_starts_goals=True,
        goals_inside=entry['family'] == 'entry_corridor',
        start_goal_height=2.,
        mouth_y=0.,
        local_interior=[-width / 2, width / 2, 0., depth - radius],
        nominal_planar_slots=nominal_slots,
        capacity_definition='safe grid at z=2, not a hard 3D occupancy limit',
        vertical_motion_restricted=False,
        portal_defined=False,
    )

    plan = LayoutPlan(spec, obstacles, starts, goals, segments, metadata)
    validate_layout(plan)
    return plan
