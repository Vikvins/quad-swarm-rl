"""Export one atomically published checkpoint and its config, without stopping training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile


def export_checkpoint(experiment, output=None):
    experiment = Path(experiment).expanduser().resolve()
    config = experiment / 'config.json'
    if not config.is_file():
        raise ValueError(f'Missing {config}; export requires the model configuration')
    pattern = re.compile(r'^checkpoint_(\d+)_(\d+)\.pth$')
    choices = [(int(match[2]), int(match[1]), path) for path in (experiment / 'checkpoint_p0').glob('checkpoint_*.pth')
               if (match := pattern.fullmatch(path.name))]
    if not choices:
        raise ValueError('No published checkpoint found yet; wait for the next save')
    steps, updates, checkpoint = max(choices)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = Path(output).expanduser() if output else Path.home() / 'Desktop' / f'{experiment.name}_{steps}_{stamp}.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    # The training code publishes checkpoint_* by atomic rename. Opening the
    # selected file first also protects its bytes against later checkpoint rotation.
    with checkpoint.open('rb') as source:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.partial', delete=False) as temp:
            temporary = Path(temp.name)
        digest = hashlib.sha256()
        try:
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                with archive.open(f'{experiment.name}/checkpoint_p0/{checkpoint.name}', 'w', force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                        target.write(chunk)
                archive.write(config, f'{experiment.name}/config.json')
                manifest = dict(checkpoint=checkpoint.name, env_steps=steps, train_step=updates,
                                sha256=digest.hexdigest(), exported_at_utc=stamp,
                                note='Only the latest saved checkpoint is exported; in-memory unsaved updates are not included.')
                archive.writestr(f'{experiment.name}/export_manifest.json', json.dumps(manifest, indent=2))
            if output.exists():
                raise FileExistsError(f'Export already exists: {output}')
            temporary.rename(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(export_checkpoint(args.experiment_dir, args.output))


if __name__ == '__main__':
    main()
