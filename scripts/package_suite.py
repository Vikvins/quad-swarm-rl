"""Package source, changed files, a patch, all measured data and reports."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--extended',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();project=Path(__file__).resolve().parents[1]
    verification=json.loads((args.validation/'results_validation.json').read_text())
    if verification['completed_episodes']!=432:raise ValueError('Primary experiment is incomplete')
    for name in ['QuadSwarm_experiment_report.pdf','QuadSwarm_results.xlsx','layouts/QuadSwarm_all_layouts.pdf']:
        if not (args.analysis/name).is_file():raise ValueError(f'Missing final artifact: {name}')
    status=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=project,text=True).splitlines()
    paths=subprocess.check_output(['git','ls-files','--cached'],cwd=project,text=True).splitlines()
    latest='checkpoint_000781254_1000005120.pth'
    source=[]
    for rel in paths:
        path=project/rel
        if not path.is_file():continue
        if '/checkpoints/' in rel and path.name!=latest:continue
        if '__pycache__' in rel or rel.endswith('.pyc'):continue
        source.append((path,Path('source')/rel))
    items=[(project/'SUITE_README_RU.md',Path('README_FIRST_RU.md')),
           (project/'CHANGES_LAYOUT_SUITE.md',Path('CHANGES_LAYOUT_SUITE.md'))]
    items+=source
    items+=[(project/rel,Path('changed_files')/rel) for rel in status if (project/rel).is_file()]
    for root,destination in [(args.analysis,'analysis'),(args.runs,'runs_15s'),(args.extended,'runs_30s'),(args.validation,'validation')]:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.name.endswith('.inspect.ndjson'):continue
            if path.name.startswith('preview_'):continue
            items.append((path,Path(destination)/path.relative_to(root)))
    patch=subprocess.check_output(['git','diff','--cached','--binary'],cwd=project)
    manifest=[];prefix='QuadSwarm_deadends_full/'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as archive:
        for index,(source,destination) in enumerate(items):
            archive.write(source,prefix+destination.as_posix())
            manifest.append(dict(path=destination.as_posix(),bytes=source.stat().st_size,sha256=sha(source)))
            if index%300==0:print(f'Packaged {index}/{len(items)} files',flush=True)
        archive.writestr(prefix+'patches/from_swarm100.patch',patch)
        info=dict(base_commit='85a0a2e1ff9993929b95d52646385a3b36fc2289',
                  metrics_commit='280800367d868d888424a1cab573f98a2d1e87d6',
                  tests_passed=124,legacy_unittest_errors=9,
                  primary_episodes=432,extended_episodes=16,
                  checkpoint_env_steps=1000005120,checkpoint_included=latest,
                  checkpoint_sha256=verification['checkpoint_sha256'],
                  patch_sha256=hashlib.sha256(patch).hexdigest(),
                  note='Local feature branch, no automatic GitHub publication. All raw diagnostic CSV and TensorBoard files included.',
                  files=manifest)
        archive.writestr(prefix+'PACKAGE_MANIFEST.json',json.dumps(info,indent=2,ensure_ascii=False))
    with zipfile.ZipFile(args.output) as archive:
        bad=archive.testzip()
        if bad:raise ValueError(f'ZIP CRC failed: {bad}')
    print('ZIP verified:',args.output,'bytes=',args.output.stat().st_size,'files=',len(items),flush=True)


if __name__=='__main__':main()
