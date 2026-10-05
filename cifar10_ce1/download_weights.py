"""Download GitHub Release assets and verify their recorded SHA256 checksums."""
import argparse
import json
from pathlib import Path
import urllib.request

from .common import sha256


def main():
    manifest = json.loads(Path(__file__).with_name('weights.json').read_text())
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('./weights'))
    p.add_argument('--all',action='store_true',help='Include FM/full resume states and evaluation scorer')
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name,asset in manifest['assets'].items():
        if not args.all and name != 'cifar10-ce1-1050000-ema.pt':
            continue
        dest = args.output/name
        if dest.exists() and sha256(dest) == asset['sha256']:
            print('Verified existing',dest)
            continue
        tmp = dest.with_suffix('.pt.part')
        print('Downloading',asset['url'],flush=True)
        urllib.request.urlretrieve(asset['url'],tmp)
        if tmp.stat().st_size != asset['bytes'] or sha256(tmp) != asset['sha256']:
            tmp.unlink()
            raise RuntimeError('Checksum mismatch: '+name)
        tmp.replace(dest)
        print('Verified',dest,flush=True)


if __name__ == '__main__':
    main()
