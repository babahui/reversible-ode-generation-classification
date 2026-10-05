"""Matched EMA reverse-CE diagnostic, without changing training or model inputs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from .common import integrate, load_model, seed_all, sha256, write_json


def balanced_order(targets, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(targets == c)) for c in range(10)]
    return np.stack(groups, axis=1).reshape(-1)


def summarize(arrays):
    logits = torch.from_numpy(arrays['center_logits'])
    labels = torch.from_numpy(arrays['targets'])
    ce = F.cross_entropy(logits, labels, reduction='none').numpy()
    other = logits.clone()
    other[torch.arange(len(labels)), labels] = -float('inf')
    margins = (logits[torch.arange(len(labels)), labels] - other.max(1).values).numpy()
    return dict(count=len(labels), correct=int((arrays['predictions'] == arrays['targets']).sum()),
                accuracy=float((arrays['predictions'] == arrays['targets']).mean()),
                center_ce_mean=float(ce.mean()), center_ce_median=float(np.median(ce)),
                center_ce_p95=float(np.quantile(ce, .95)),
                center_margin_mean=float(margins.mean()), center_margin_median=float(np.median(margins)),
                posterior_nll_mean=float(arrays['posterior_nll'].mean()))


def paired(before, after):
    if not np.array_equal(before['indices'], after['indices']):
        raise ValueError('Unmatched images')
    if not np.array_equal(before['targets'], after['targets']):
        raise ValueError('Unmatched targets')
    a = before['predictions'] == before['targets']
    b = after['predictions'] == after['targets']
    return dict(count=len(a), wrong_to_right=int((~a & b).sum()),
                right_to_wrong=int((a & ~b).sum()), both_wrong=int((~a & ~b).sum()),
                prediction_disagreement=float((before['predictions'] != after['predictions']).mean()))


def evaluate(model, prior, dataset, indices, method, steps, batch_size, output, progress):
    if output.exists():
        with np.load(output) as stored:
            arrays = dict(stored)
        if not np.array_equal(arrays['indices'], indices):
            raise ValueError('Cached subset mismatch')
        return arrays
    collected = {k: [] for k in ('indices', 'targets', 'predictions', 'center_logits', 'posterior_nll')}
    tic = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            x = torch.from_numpy(dataset.data[ids]).permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
            labels = torch.tensor(np.asarray(dataset.targets)[ids], device='cuda')
            encoded, _ = integrate(model, x, None, method, steps, reverse=True)
            center = prior.center_coordinates(encoded)
            scores = prior.log_posterior(encoded)
            if not bool(torch.isfinite(center).all() and torch.isfinite(scores).all()):
                raise RuntimeError('Nonfinite endpoint logits')
            values = dict(indices=ids, targets=labels.cpu().numpy(),
                          predictions=scores.argmax(1).cpu().numpy(), center_logits=center.cpu().numpy(),
                          posterior_nll=F.cross_entropy(scores, labels, reduction='none').cpu().numpy())
            for key, value in values.items():
                collected[key].append(value)
            state = dict(job=output.stem, evaluated=start + len(ids), total=len(indices),
                         seconds=time.monotonic() - tic)
            write_json(progress, state)
            if start == 0 or (start // batch_size + 1) % 10 == 0:
                print(json.dumps(state), flush=True)
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    tmp = output.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(output)
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--finetuned', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--subset', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260917)
    args = parser.parse_args()
    if args.subset < 10 or args.subset > 10000 or args.subset % 10:
        raise ValueError('Use a balanced subset of 10..10000 images')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(args.seed)
    datasets = {split: CIFAR10(args.data_root, train=split == 'train', download=False)
                for split in ('train', 'test')}
    orders = {split: balanced_order(np.asarray(data.targets), args.seed) for split, data in datasets.items()}
    protocol = dict(checkpoints={key: dict(path=str(Path(path).resolve()), sha256=sha256(path))
                                 for key, path in [('baseline', args.baseline), ('ce50k', args.finetuned)]},
                    seed=args.seed, subset=args.subset, batch_size=args.batch_size,
                    normalization='uint8 / 127.5 - 1', weights='EMA', dtype='float32', tf32=False,
                    augmentation=False, label_input=False, solvers=['euler4', 'heun60'],
                    full_test_euler4=True, data_root=str(Path(args.data_root).resolve()))
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing diagnostic uses a different protocol')
    write_json(protocol_path, protocol)
    results, all_arrays = {}, {}
    for name, checkpoint in [('baseline', args.baseline), ('ce50k', args.finetuned)]:
        model, config, prior, step = load_model(checkpoint)
        if prior is None or config['kind'] not in ('gaussian', 'gaussian_ce'):
            raise ValueError('Expected Gaussian-only or Gaussian+CE model')
        results[name] = dict(step=step, metrics={})
        all_arrays[name] = {}
        for split in ('train', 'test'):
            for method, steps in [('euler', 4), ('heun', 60)]:
                ids = orders[split] if split == 'test' and method == 'euler' else orders[split][:args.subset]
                key = '{}_{}{}'.format(split, method, steps)
                arrays = evaluate(model, prior, datasets[split], ids, method, steps, args.batch_size,
                                  out / '{}_{}.npz'.format(name, key), out / 'progress.json')
                results[name]['metrics'][key] = summarize(arrays)
                all_arrays[name][key] = arrays
                print(json.dumps(dict(model=name, job=key, **results[name]['metrics'][key])), flush=True)
                write_json(out / 'metrics.json', results)
        del model, prior
        torch.cuda.empty_cache()
    comparisons = dict(finetune={}, solvers={})
    for key in all_arrays['baseline']:
        comparisons['finetune'][key] = paired(all_arrays['baseline'][key], all_arrays['ce50k'][key])
    for name in all_arrays:
        comparisons['solvers'][name] = {}
        for split in ('train', 'test'):
            a = {k: v[:args.subset] for k, v in all_arrays[name][split + '_euler4'].items()}
            b = all_arrays[name][split + '_heun60']
            comparisons['solvers'][name][split] = dict(euler=summarize(a), heun=summarize(b),
                                                       euler_to_heun=paired(a, b))
    write_json(out / 'paired.json', comparisons)
    write_json(out / 'progress.json', dict(phase='complete'))


if __name__ == '__main__':
    main()
