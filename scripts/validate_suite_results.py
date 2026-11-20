"""Validate completeness and paired geometry of a finished benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def lines(path):
    total=0
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):total+=chunk.count(b'\n')
    return total-1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.input
    manifest=json.loads((root/'manifest.json').read_text())
    checkpoint=Path(manifest['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint bytes changed')
    project=Path(__file__).resolve().parents[1]
    for rel,expected_sha in manifest['source_sha256'].items():
        if hashlib.sha256((project/rel).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError(f'Executed source changed: {rel}')
    config=project/'experiments/swarm100-1b/config.json'
    if hashlib.sha256(config.read_bytes()).hexdigest()!=manifest['original_config_sha256']:
        raise ValueError('Original model config changed')
    from gym_art.quadrotor_multi.obstacles.suite_layouts import catalogue,FAMILIES
    expected={(n,r['id'],seed) for n,count in [(8,16),(100,8)] for r in catalogue() for seed in range(count)}
    records=[];actual=set();geometry={};total_metric_rows=total_influence_rows=0
    for file in sorted(root.glob('n*/*/seed*/episode.json')):
        r=json.loads(file.read_text());key=(r['agents'],r['layout_id'],r['seed'])
        if key in actual:raise ValueError('Duplicate episode')
        actual.add(key)
        if r['steps']!=1501 or not np.isclose(r['duration'],15.01):raise ValueError('Incomplete horizon')
        folder=file.parent
        plan=json.loads((folder/'layout.json').read_text());trajectory=np.load(folder/'trajectory.npz')['positions']
        if trajectory.shape!=(1502,r['agents'],3) or not np.isfinite(trajectory).all():raise ValueError('Invalid trajectory')
        np.testing.assert_allclose(trajectory[0],plan['starts'],atol=1e-6)
        m=lines(folder/'diagnostics/metrics.csv');i=lines(folder/'diagnostics/influence.csv')
        if m!=1501*r['agents'] or i!=151*r['agents']:raise ValueError('Truncated raw diagnostics')
        total_metric_rows+=m;total_influence_rows+=i
        geometry[key]=(np.asarray(plan['starts']),np.asarray(plan['goals']))
        records.append(dict(agents=r['agents'],layout_id=r['layout_id'],seed=r['seed'],metrics_rows=m,influence_rows=i))
    if actual!=expected:raise ValueError(f'Unexpected/missing episodes: {len(actual)} vs {len(expected)}')
    pairs=0
    for n,count in [(8,16),(100,8)]:
        for family in FAMILIES:
            for seed in range(count):
                a=geometry[n,family+'_open',seed];b=geometry[n,family+'_deadend',seed]
                np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);pairs+=1
    result=dict(completed_episodes=len(actual),agent_episodes=sum(n for n,_,_ in actual),
                matched_pairs=pairs,raw_metrics_rows=total_metric_rows,raw_influence_rows=total_influence_rows,
                trajectories_finite=True,paired_starts_goals_identical=True,checkpoint_unchanged=True,
                executed_source_hashes_match=True,source_files_checked=len(manifest['source_sha256']),
                original_config_unchanged=True,
                checkpoint_sha256=manifest['checkpoint_sha256'],records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
