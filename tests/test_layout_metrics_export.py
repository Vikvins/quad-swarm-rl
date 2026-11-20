import json
import zipfile

import numpy as np
import pytest

from gym_art.quadrotor_multi.obstacles.layouts import LayoutSpec, build_layout
from swarm_rl.layout_metrics import LayoutMetrics
from swarm_rl.tools.export_checkpoint import export_checkpoint


def test_contacts_during_initial_grace_period_still_count():
    plan = build_layout(LayoutSpec('corridor', num_agents=1))
    metrics = LayoutMetrics(plan, .01, .092)
    metrics.update(plan.obstacles[:1], [False])
    assert metrics.summary()['obstacle_collision_rate'] == 1
    assert metrics.summary()['success_rate'] == 0


def test_goal_requires_hold_and_collision_free_episode():
    plan = build_layout(LayoutSpec('corridor', num_agents=1))
    metrics = LayoutMetrics(plan, .1, .092, goal_hold=.5)
    for _ in range(4):
        metrics.update(plan.goals, [False])
    assert metrics.summary()['success_rate'] == 0
    metrics.update(plan.goals, [False])
    assert metrics.summary()['success_rate'] == 1
    metrics.update(plan.goals, [True])
    assert metrics.summary()['success_rate'] == 0


def test_neighbor_collision_and_stall_are_not_goal_success():
    plan = build_layout(LayoutSpec('corridor', num_agents=8))
    metrics = LayoutMetrics(plan, .1, .092)
    positions = plan.starts.copy()
    positions[1] = positions[0]
    for _ in range(31):
        metrics.update(positions, np.zeros(8, bool))
    rows = metrics.agent_rows()
    assert rows[0]['neighbor_collision'] == rows[1]['neighbor_collision'] == 1
    assert rows[0]['stalled'] == 1
    assert rows[0]['success'] == 0


def test_export_selects_latest_saved_steps_and_preserves_config(tmp_path):
    experiment = tmp_path / 'paper_attention_seed0'
    checkpoints = experiment / 'checkpoint_p0'
    checkpoints.mkdir(parents=True)
    config = json.dumps({'experiment': experiment.name, 'train_for_env_steps': 1000000000})
    (experiment / 'config.json').write_text(config)
    (checkpoints / 'checkpoint_000000010_10000.pth').write_bytes(b'first')
    (checkpoints / 'checkpoint_000000011_11000.pth').write_bytes(b'latest')
    (checkpoints / 'checkpoint_temp.pth').write_bytes(b'in-progress')
    output = tmp_path / 'export.zip'
    export_checkpoint(experiment, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read(experiment.name + '/config.json').decode() == config
        assert archive.read(experiment.name + '/checkpoint_p0/checkpoint_000000011_11000.pth') == b'latest'
        manifest = json.loads(archive.read(experiment.name + '/export_manifest.json'))
        assert manifest['env_steps'] == 11000
        assert len(archive.namelist()) == 3
    assert (checkpoints / 'checkpoint_temp.pth').read_bytes() == b'in-progress'
    with pytest.raises(FileExistsError):
        export_checkpoint(experiment, output)
    assert output.is_file()


def test_export_without_checkpoint_fails_clearly(tmp_path):
    experiment = tmp_path / 'empty'
    experiment.mkdir()
    with pytest.raises(ValueError):
        export_checkpoint(experiment)
    (experiment / 'config.json').write_text('{}')
    with pytest.raises(ValueError):
        export_checkpoint(experiment)
