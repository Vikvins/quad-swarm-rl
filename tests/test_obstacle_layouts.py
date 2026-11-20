from dataclasses import replace

import numpy as np
import pytest

from gym_art.quadrotor_multi.obstacles.layouts import (
    LayoutSpec, build_layout, navigation_grid, obstacle_clearance, reference_route, segment_clearance,
)


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
@pytest.mark.parametrize('agents', [1, 8, 100])
@pytest.mark.parametrize('width', [.8, 1.2, 1.6])
def test_layouts_are_traversable_and_start_without_contacts(kind, agents, width):
    plan = build_layout(LayoutSpec(kind, num_agents=agents, width=width))
    route = reference_route(plan)
    assert route.shape[1] == 2
    for p, q in zip(route, route[1:]):
        assert segment_clearance(p, q, plan.obstacles, .3) >= .146 - 1e-8
    for points in [plan.starts, plan.goals]:
        assert obstacle_clearance(points[:, :2], plan.obstacles, .3).min() >= .146
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= .3 - 1e-8


@pytest.mark.parametrize('kind', ['corridor', 'dead_end'])
@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_rotations_remain_reachable(kind, rotation):
    plan = build_layout(LayoutSpec(kind, rotation=rotation, num_agents=100))
    assert reference_route(plan).size


def test_seed_reproducibility_and_no_global_rng_side_effects():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    a = build_layout(LayoutSpec('dead_end', seed=42))
    np.testing.assert_array_equal(np.random.random(3), expected)
    b = build_layout(LayoutSpec('dead_end', seed=42))
    c = build_layout(LayoutSpec('dead_end', seed=43))
    np.testing.assert_array_equal(a.obstacles, b.obstacles)
    np.testing.assert_array_equal(a.starts, b.starts)
    np.testing.assert_array_equal(a.goals, b.goals)
    assert not np.array_equal(a.starts, c.starts)


def test_corridor_cannot_be_bypassed_outside_side_walls():
    plan = build_layout(LayoutSpec('corridor', num_agents=1))
    a = plan.spec.width / 2 + .3
    barrier = np.column_stack((np.linspace(-a, a, 15), np.zeros(15), np.full(15, 5.)))
    sealed = replace(plan, obstacles=np.concatenate((plan.obstacles, barrier)))
    # A barrier across the corridor must disconnect start and goal, even though
    # the side walls have finite length and there is room on either side.
    with pytest.raises(ValueError):
        reference_route(sealed)


def test_dead_end_requires_retreat_and_goals_are_not_impossible():
    plan = build_layout(LayoutSpec('dead_end', num_agents=8))
    route = reference_route(plan)
    mouth = -plan.spec.dead_end_depth / 2
    assert np.all(plan.starts[:, 1] > mouth)
    assert np.all(plan.goals[:, 1] > plan.spec.dead_end_depth / 2 + .3)
    assert route[:, 1].min() < mouth - .3
    a = plan.spec.width / 2 + .3
    cap = np.column_stack((np.linspace(-a, a, 15), np.full(15, mouth), np.full(15, 5.)))
    with pytest.raises(ValueError):
        reference_route(replace(plan, obstacles=np.concatenate((plan.obstacles, cap))))


def test_physics_sdf_and_collision_geometry_match_generated_map():
    from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection
    plan = build_layout(LayoutSpec('corridor', num_agents=1))
    positions = np.array([[0., 0.], [.9, 0.], [.55, 0.], [3., -2.5]])
    actual = get_surround_sdfs(positions, plan.obstacles[:, :2], np.full((len(positions), 9), 100.), .3)
    for i, point in enumerate(positions):
        probes = np.array([[point[0] + dx, point[1] + dy]
                           for dx in [-.1, 0., .1] for dy in [-.1, 0., .1]])
        expected = obstacle_clearance(probes, plan.obstacles, .3)
        np.testing.assert_allclose(actual[i], expected, atol=1e-10)
    collided = collision_detection(positions, plan.obstacles[:, :2], .3, .046) >= 0
    np.testing.assert_array_equal(collided, obstacle_clearance(positions, plan.obstacles, .3) <= .046)


@pytest.mark.parametrize('updates', [dict(wall_spacing=.7), dict(width=.2), dict(num_agents=100000),
                                    dict(spawn_separation=.05), dict(corridor_length=20.),
                                    dict(rotation=45), dict(width=float('nan'))])
def test_invalid_layouts_fail_instead_of_silently_making_impossible_tasks(updates):
    with pytest.raises(ValueError):
        build_layout(LayoutSpec('corridor', **updates))
