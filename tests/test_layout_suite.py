from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from gym_art.quadrotor_multi.obstacles.layouts import LayoutSpec, build_layout, reference_route
from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue,FAMILIES
from swarm_rl.run_layout_suite import make_suite_config


@pytest.mark.parametrize('layout_id',[r['id'] for r in catalogue()[1:]])
@pytest.mark.parametrize('agents',[8,100])
def test_all_suite_maps_are_reachable_and_nonoverlapping(layout_id,agents):
    plan=build_layout(LayoutSpec('suite',layout_id=layout_id,num_agents=agents,room_dims=(14.,14.,10.)))
    assert reference_route(plan).shape[0]>1
    assert plan.starts.shape==(agents,3)
    assert plan.metadata['id']==layout_id


@pytest.mark.parametrize('family',list(FAMILIES))
@pytest.mark.parametrize('seed',[0,1,7])
def test_paired_maps_have_identical_starts_goals_and_added_walls_only(family,seed):
    a=build_layout(LayoutSpec('suite',layout_id=family+'_open',num_agents=100,room_dims=(14.,14.,10.),seed=seed))
    b=build_layout(LayoutSpec('suite',layout_id=family+'_deadend',num_agents=100,room_dims=(14.,14.,10.),seed=seed))
    np.testing.assert_array_equal(a.starts,b.starts)
    np.testing.assert_array_equal(a.goals,b.goals)
    aset={tuple(np.round(row,8)) for row in a.obstacles}
    bset={tuple(np.round(row,8)) for row in b.obstacles}
    assert aset < bset


@pytest.mark.parametrize('family',['straight','narrow','elbow','zigzag','funnel','wide_u','comb'])
def test_capped_passage_requires_retreat_through_entrance(family):
    plan=build_layout(LayoutSpec('suite',layout_id=family+'_deadend',num_agents=8,room_dims=(14.,14.,10.)))
    mouth=-2. if family=='wide_u' else -5.
    route=reference_route(plan)
    assert route[:,1].min()<mouth-.3


def test_suite_uses_original_training_config_and_disables_replay():
    original=json.loads(Path('experiments/swarm100-1b/config.json').read_text())
    cfg=make_suite_config(original,'straight_deadend',100,15.)
    assert cfg.quads_room_dims==[14.,14.,10.]
    assert cfg.quads_obst_spawn_area==[12.,12.]
    assert cfg.quads_neighbor_visible_num==2
    assert cfg.quads_use_downwash is True
    assert cfg.replay_buffer_sample_prob==0
    assert cfg.device=='cpu'
    assert cfg.eval_deterministic
    assert cfg.layout_id=='straight_deadend'


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_obs_dim=19
        self.all_neighbor_obs_dim=12
        self.weights=torch.arange(160,dtype=torch.float32).reshape(40,4)/50000
        self.last_action_distribution=None
    def forward(self,obs,states):
        means=obs['obs'] @ self.weights
        self.last_action_distribution=SimpleNamespace(means=means)
        return {'actions':means}
    def action_distribution(self):
        return self.last_action_distribution


def test_influence_equations_match_independent_affine_reference_and_preserve_actions(tmp_path,monkeypatch):
    from drone_influence import InfluenceMetrics
    monkeypatch.setenv('METRICS_TB_AGENTS','0')
    model=FakePolicy()
    obs={'obs':torch.linspace(-.2,.2,80).reshape(2,40)}
    states=torch.zeros(2,1)
    model(obs,states)
    before=model.action_distribution()
    cfg=SimpleNamespace(device='cpu',use_rnn=False,quads_obs_repr='xyz_vxyz_R_omega_floor',
                        normalize_input=False,obs_scale=1.,obs_subtract_mean=0.,env_frameskip=1,
                        eval_env_frameskip=1,eval_deterministic=True)
    env=SimpleNamespace(action_space=SimpleNamespace(low=-np.ones(4),high=np.ones(4)),control_freq=100.)
    metrics=InfluenceMetrics(tmp_path/'metrics',.02,8,10,None)
    try:
        metrics.observe(cfg,env,model,obs,states)
        assert model.action_distribution() is before
        effects=[]
        for a,b in [(19,31),(31,40)]:
            change=(obs['obs'][:,a:b] @ model.weights[a:b]).numpy()
            effects.append(.01*.5*np.sqrt(np.mean(change**2,axis=1)))
        np.testing.assert_allclose(metrics.last_influence['neighbors_deviation'],effects[0],rtol=2e-4,atol=1e-8)
        np.testing.assert_allclose(metrics.last_influence['obstacles_deviation'],effects[1],rtol=2e-4,atol=1e-8)
        expected=effects[1]/(effects[0]+effects[1])
        np.testing.assert_allclose(metrics.last_influence['obstacle_share'],expected,rtol=2e-4)
        rng=np.random.default_rng(2026)
        perturbations=[]
        for _ in range(8):
            delta=rng.normal(0,.02,size=(2,2))
            perturbations.append(np.mean((delta @ model.weights[:2].numpy()/2)**2,axis=1))
        np.testing.assert_allclose(metrics.last_arrays['sensitivity_xy'],np.sqrt(np.mean(perturbations,axis=0)),rtol=2e-3,atol=1e-8)
        metrics.observe(cfg,env,model,obs,states)
        assert np.all(metrics.last_arrays['command_change_rate']==0)
        assert np.all(metrics.last_arrays['saturation_fraction']==0)
    finally:
        metrics.close()
