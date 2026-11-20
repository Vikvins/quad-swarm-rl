"""Download the verified 1B-step checkpoint without cloning every Git LFS object."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

NAME = 'checkpoint_000781254_1000005120.pth'
COMMIT = '02a68525ec250461d8058f7cf40e2b264b9d859b'
URL = f'https://media.githubusercontent.com/media/YaroslavPelekhov/quad-swarm-experiments/{COMMIT}/checkpoints/{NAME}'
SHA256 = 'f429fa60b70a82ba1aaf54ebbdd0f966d65995d46caf87e0fd8628e1f15b2524'
SIZE = 39718410


def download(directory):
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / NAME
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
            return destination
        raise ValueError(f'{destination} already exists but has a different hash; choose a new output directory')
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.partial', delete=False) as temp:
        temporary = Path(temp.name)
        digest = hashlib.sha256()
        count = 0
        try:
            request = urllib.request.Request(URL, headers={'User-Agent': 'QuadSwarm-layout-evaluation'})
            with urllib.request.urlopen(request, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    temp.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            temp.flush()
            if count != SIZE or digest.hexdigest() != SHA256:
                raise ValueError('Downloaded bytes do not match the verified checkpoint; file was not installed')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.home() / 'Desktop/QuadSwarm-pretrained')
    args = parser.parse_args()
    path = download(args.output)
    print('Downloaded/verified:', path)
    print('Environment steps: 1,000,005,120')
    print('SHA256:', SHA256)
    print('Original config.json is not present in the weights repository.')


if __name__ == '__main__':
    main()
