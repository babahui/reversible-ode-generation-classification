"""Clean full-test reverse classification with saved per-image predictions."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from .common import load_model, sha256, write_json
from .diagnose_ce_solvers import evaluate, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--method', choices=['euler', 'heun'], default='euler')
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--split', choices=['train','test'], default='test')
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=sha256(args.checkpoint), method=args.method, steps=args.steps,
                    batch_size=64, count=50000 if args.split=='train' else 10000, split=args.split, weights='EMA', dtype='float32', tf32=False,
                    augmentation=False, label_input=False, data_root=str(Path(args.data_root).resolve()))
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('Existing reverse output protocol mismatch')
    write_json(out / 'protocol.json', protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config, prior, step = load_model(args.checkpoint)
    dataset = CIFAR10(args.data_root, train=args.split=='train', download=False)
    arrays = evaluate(model, prior, dataset, np.arange(len(dataset)), args.method, args.steps, 64,
                      out / 'predictions.npz', out / 'progress.json')
    if len(dataset) != protocol['count'] or not np.array_equal(arrays['targets'], dataset.targets):
        raise ValueError('Unexpected test data')
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (arrays['targets'], arrays['predictions']), 1)
    metrics = dict(summarize(arrays), checkpoint_step=step, complete=True,
                   confusion_matrix=confusion.tolist(), accuracy_per_class=(confusion.diagonal()/(5000 if args.split=='train' else 1000)).tolist())
    write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
