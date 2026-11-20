"""Prepare a standard enjoy experiment directory from the published 1B model."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=root/'experiments/swarm100-1b/checkpoints/checkpoint_000781254_1000005120.pth')
    p.add_argument('--config',type=Path,default=root/'experiments/swarm100-1b/config.json')
    p.add_argument('--train-dir',type=Path,default=root/'train_dir')
    p.add_argument('--experiment',default='swarm100-1b-eval')
    args=p.parse_args()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size<1024:
        p.error('A real checkpoint, not a Git LFS pointer, is required')
    target=args.train_dir/args.experiment
    target.mkdir(parents=True,exist_ok=False)
    (target/'checkpoint_p0').mkdir()
    shutil.copy2(args.checkpoint,target/'checkpoint_p0'/args.checkpoint.name)
    cfg=json.loads(args.config.read_text())
    cfg.update(experiment=args.experiment,train_dir=str(args.train_dir.resolve()))
    (target/'config.json').write_text(json.dumps(cfg,indent=2))
    shutil.copy2(args.config,target/'original_training_config.json')
    print('Prepared:',target.resolve())


if __name__=='__main__':main()
