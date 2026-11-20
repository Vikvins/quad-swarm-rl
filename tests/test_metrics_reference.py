import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_layout_suite import FakePolicy


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(cls,folder):
    model=FakePolicy()
    model.all_neighbor_obs_size=model.all_neighbor_obs_dim
    cfg=SimpleNamespace(device='cpu',use_rnn=False,quads_obs_repr='xyz_vxyz_R_omega_floor',normalize_input=False,
                        obs_scale=1.,obs_subtract_mean=0.,env_frameskip=1,eval_env_frameskip=1,eval_deterministic=True)
    env=SimpleNamespace(action_space=SimpleNamespace(low=-np.ones(4),high=np.ones(4)),control_freq=100.)
    metric=cls(folder,.02,8,1,None)
    try:
        for step in range(4):
            obs={'obs':torch.linspace(-.2,.2,80).reshape(2,40)+step*.01}
            states=torch.zeros(2,1);model(obs,states)
            metric.observe(cfg,env,model,obs,states)
    finally:metric.close()


def test_metrics_csv_is_identical_to_supplied_script(tmp_path,monkeypatch):
    from drone_metrics import Metrics
    original=load('reference_metrics',Path('references/drone_metrics.original.py'))
    monkeypatch.setenv('METRICS_TB_AGENTS','0')
    run(original.Metrics,tmp_path/'original');run(Metrics,tmp_path/'new')
    assert (tmp_path/'original/metrics.csv').read_bytes()==(tmp_path/'new/metrics.csv').read_bytes()


def test_influence_csv_is_identical_to_supplied_formula(tmp_path,monkeypatch):
    from drone_influence import InfluenceMetrics
    original=load('reference_influence',Path('references/drone_influence.original.py'))
    monkeypatch.setenv('METRICS_TB_AGENTS','0')
    run(original.InfluenceMetrics,tmp_path/'original');run(InfluenceMetrics,tmp_path/'new')
    assert (tmp_path/'original/influence.csv').read_bytes()==(tmp_path/'new/influence.csv').read_bytes()
    assert (tmp_path/'original/metrics.csv').read_bytes()==(tmp_path/'new/metrics.csv').read_bytes()
