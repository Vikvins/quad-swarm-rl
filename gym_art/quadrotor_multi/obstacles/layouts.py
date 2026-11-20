"""Deterministic benchmarks built from the simulator's existing cylinder geometry."""

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class LayoutSpec:
    kind: str
    num_agents: int = 8
    room_dims: tuple = (10., 10., 10.)
    width: float = 1.2
    corridor_length: float = 5.
    dead_end_depth: float = 3.
    obstacle_size: float = .6
    wall_spacing: float = .45
    agent_radius: float = .046
    safety_margin: float = .1
    spawn_separation: float = .3
    z_min: float = 1.
    z_max: float = 3.
    rotation: int = 0
    seed: int = 0
    grid_resolution: float = .1
    layout_id: str = ''


@dataclass
class LayoutPlan:
    spec: LayoutSpec
    obstacles: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    wall_segments: list
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(spec=asdict(self.spec), obstacles=self.obstacles.tolist(),
                    starts=self.starts.tolist(), goals=self.goals.tolist(),
                    wall_segments=self.wall_segments, metadata=self.metadata)


def _axis(low, high, separation):
    if high < low:
        raise ValueError('The requested layout leaves no space for a spawn/goal zone')
    count = int(np.floor((high - low + 1e-10) / separation)) + 1
    return np.linspace(low, high, max(1, count))


def _zone(x_range, y_range, spec, rng):
    axes = [_axis(*x_range, spec.spawn_separation),
            _axis(*y_range, spec.spawn_separation),
            _axis(spec.z_min, spec.z_max, spec.spawn_separation)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    if len(grid) < spec.num_agents:
        raise ValueError(f'Spawn/goal zone has capacity {len(grid)}, requested {spec.num_agents}. '
                         'Increase room/zone size or reduce agent count; spacing is never silently reduced.')
    return grid[rng.permutation(len(grid))[:spec.num_agents]]


def _wall(start, end, spacing):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    return np.linspace(start, end, max(2, int(np.ceil(length / spacing)) + 1))


def build_layout(spec):
    if spec.kind == 'suite':
        from gym_art.quadrotor_multi.obstacles.suite_layouts import build_suite_layout
        return build_suite_layout(spec)
    if spec.kind not in ('corridor', 'dead_end'):
        raise ValueError(f'Unknown layout: {spec.kind}')
    values = [spec.width, spec.corridor_length, spec.dead_end_depth, spec.obstacle_size,
              spec.wall_spacing, spec.agent_radius, spec.spawn_separation, spec.grid_resolution,
              *spec.room_dims, spec.z_min, spec.z_max, spec.safety_margin]
    if not np.isfinite(values).all() or min(values) <= 0 or spec.num_agents < 1:
        raise ValueError('Layout dimensions, radii, margins and agent count must be positive and finite')
    if spec.wall_spacing > spec.obstacle_size:
        raise ValueError('Wall spacing exceeds cylinder diameter: a wall would have holes')
    if spec.rotation not in (0, 90, 180, 270):
        raise ValueError('Only 0/90/180/270 degree rotations are supported')
    if spec.rotation in (90, 270) and not np.isclose(spec.room_dims[0], spec.room_dims[1]):
        raise ValueError('Quarter-turn rotations currently require a square room')
    if spec.spawn_separation < 2 * spec.agent_radius + spec.safety_margin:
        raise ValueError('Spawn separation is too small for the drone collision geometry')
    pad = spec.agent_radius + spec.safety_margin + spec.grid_resolution
    if spec.width <= 2 * pad:
        raise ValueError('Corridor is too narrow for the requested clearance and grid validation')
    if spec.z_min < max(.75, pad) or spec.z_max < spec.z_min or spec.z_max > spec.room_dims[2] - pad:
        raise ValueError('Spawn heights are outside the safe room interior')
    hx, hy = np.asarray(spec.room_dims[:2]) / 2
    radius = spec.obstacle_size / 2
    a = spec.width / 2 + radius
    if a + radius + 2 * pad >= hx:
        raise ValueError('Not enough room outside the walls for a bypass route')
    rng = np.random.default_rng(spec.seed)
    if spec.kind == 'corridor':
        b = spec.corridor_length / 2
        # Wings connect both portals to room boundaries. Going around the
        # finite side walls must not be a shortcut around the benchmark.
        segments = [((-a, -b), (-a, b)), ((a, -b), (a, b)),
                    ((-hx, -b), (-a, -b)), ((a, -b), (hx, -b)),
                    ((-hx, b), (-a, b)), ((a, b), (hx, b))]
        starts = _zone((-hx + pad, hx - pad), (-hy + pad, -b - radius - pad), spec, rng)
        goals = _zone((-hx + pad, hx - pad), (b + radius + pad, hy - pad), spec, rng)
    else:
        b = spec.dead_end_depth / 2
        segments = [((-a, -b), (-a, b)), ((a, -b), (a, b)), ((-a, b), (a, b))]
        starts = _zone((-spec.width / 2 + pad, spec.width / 2 - pad),
                       (-b + pad, b - radius - pad), spec, rng)
        goals = _zone((-a, a), (b + radius + pad, hy - pad), spec, rng)
    xy = np.concatenate([_wall(p, q, spec.wall_spacing) for p, q in segments])
    xy = np.unique(np.round(xy, 10), axis=0)
    angle = np.deg2rad(spec.rotation)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = xy @ rotation.T
    starts[:, :2] = starts[:, :2] @ rotation.T
    goals[:, :2] = goals[:, :2] @ rotation.T
    transformed = [(np.asarray([p, q]) @ rotation.T).tolist() for p, q in segments]
    obstacles = np.column_stack((xy, np.full(len(xy), spec.room_dims[2] / 2)))
    plan = LayoutPlan(spec, obstacles, starts, goals, transformed)
    validate_layout(plan)
    return plan


def obstacle_clearance(points, obstacles, radius):
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(obstacles):
        return np.full(len(points), np.inf)
    delta = points[:, None, :] - np.asarray(obstacles)[None, :, :2]
    return np.sqrt((delta ** 2).sum(axis=-1)).min(axis=1) - radius


def _minimum_separation(points):
    if len(points) < 2:
        return np.inf
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def segment_clearance(start, end, obstacles, radius):
    start, end = np.asarray(start)[:2], np.asarray(end)[:2]
    direction = end - start
    centres = np.asarray(obstacles)[:, :2]
    if not len(centres):
        return np.inf
    t = np.clip(((centres - start) @ direction) / max(float(direction @ direction), 1e-20), 0., 1.)
    return float(np.linalg.norm(centres - start - t[:, None] * direction, axis=1).min() - radius)


def navigation_grid(plan):
    spec = plan.spec
    hx, hy = np.asarray(spec.room_dims[:2]) / 2
    step = spec.grid_resolution
    xs = np.linspace(-hx, hx, int(np.ceil(2 * hx / step)) + 1)
    ys = np.linspace(-hy, hy, int(np.ceil(2 * hy / step)) + 1)
    points = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1)
    # Extra half-diagonal ensures whole cells/4-connected edges are clear,
    # rather than merely their sample centres.
    clearance = spec.agent_radius + spec.safety_margin + step / np.sqrt(2)
    free = obstacle_clearance(points.reshape(-1, 2), plan.obstacles, spec.obstacle_size / 2)
    free = free.reshape(points.shape[:2]) >= clearance
    free &= np.abs(points[..., 0]) <= hx - clearance
    free &= np.abs(points[..., 1]) <= hy - clearance
    return xs, ys, free


def reference_route(plan):
    """Validate all start/goal connectivity; return one witness path, not a controller."""
    xs, ys, free = navigation_grid(plan)

    def index(point):
        i, j = int(np.argmin(abs(xs - point[0]))), int(np.argmin(abs(ys - point[1])))
        candidates = [(x, y) for x in range(max(0, i - 2), min(len(xs), i + 3))
                      for y in range(max(0, j - 2), min(len(ys), j + 3)) if free[x, y]]
        candidates.sort(key=lambda cell: np.linalg.norm(np.array([xs[cell[0]], ys[cell[1]]]) - point[:2]))
        for cell in candidates:
            target = np.array([xs[cell[0]], ys[cell[1]]])
            if np.linalg.norm(target - point[:2]) > 2 * plan.spec.grid_resolution:
                continue
            if segment_clearance(point, target, plan.obstacles, plan.spec.obstacle_size / 2) >= plan.spec.agent_radius + plan.spec.safety_margin - 1e-8:
                return cell
        raise ValueError('A start/goal cannot be safely connected to the navigation grid')

    starts = [index(p) for p in plan.starts]
    goals = [index(p) for p in plan.goals]
    previous = {starts[0]: None}
    pending = deque([starts[0]])
    while pending:
        i, j = pending.popleft()
        for cell in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            x, y = cell
            if 0 <= x < len(xs) and 0 <= y < len(ys) and free[cell] and cell not in previous:
                previous[cell] = (i, j)
                pending.append(cell)
    if any(cell not in previous for cell in starts + goals):
        raise ValueError('A start or goal is disconnected: the generated task is not traversable')
    path = []
    cursor = goals[0]
    while cursor is not None:
        path.append((xs[cursor[0]], ys[cursor[1]]))
        cursor = previous[cursor]
    return np.asarray(path[::-1])


def validate_layout(plan):
    spec = plan.spec
    clearance = spec.agent_radius + spec.safety_margin
    for name, points in [('starts', plan.starts), ('goals', plan.goals)]:
        if points.shape != (spec.num_agents, 3) or not np.isfinite(points).all():
            raise ValueError(f'Invalid {name}')
        if np.any(obstacle_clearance(points[:, :2], plan.obstacles, spec.obstacle_size / 2) < clearance - 1e-8):
            raise ValueError(f'{name} intersect expanded obstacles')
        half_xy = np.asarray(spec.room_dims[:2]) / 2
        if np.any(np.abs(points[:, :2]) > half_xy - clearance + 1e-8):
            raise ValueError(f'{name} outside room interior')
        if _minimum_separation(points) < spec.spawn_separation - 1e-8:
            raise ValueError(f'{name} have overlapping agents')
    route = reference_route(plan)
    return dict(route_exists=True, num_obstacles=len(plan.obstacles),
                reference_path_length=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
                min_start_clearance=float(obstacle_clearance(plan.starts[:, :2], plan.obstacles, spec.obstacle_size / 2).min()),
                min_goal_clearance=float(obstacle_clearance(plan.goals[:, :2], plan.obstacles, spec.obstacle_size / 2).min()),
                min_start_separation=_minimum_separation(plan.starts),
                min_goal_separation=_minimum_separation(plan.goals))
