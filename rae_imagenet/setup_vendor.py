"""Fetch public upstream source, pinned to the versions used in this experiment."""
import argparse
from pathlib import Path
import subprocess
from .config import RAEV2_COMMIT, DINOV3_COMMIT


def fetch(repo, commit, target):
    target = Path(target)
    if target.exists():
        current = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if current != commit:
            raise RuntimeError(f'{target} already exists at another commit; use a new destination')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', repo], check=True)
    subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', commit], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', default='third_party')
    args = p.parse_args()
    fetch('https://github.com/nanovisionx/RAEv2.git', RAEV2_COMMIT, Path(args.directory) / 'RAEv2')
    fetch('https://github.com/facebookresearch/dinov3.git', DINOV3_COMMIT, Path(args.directory) / 'dinov3')


if __name__ == '__main__':
    main()
