"""EMA generation or label-free reverse classification, without Inception."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from .common import initial_noise, integrate, load_model, sha256, to_uint8, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--method', choices=['heun', 'euler'], default='heun')
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--per-class', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--reverse-data-root')
    parser.add_argument('--limit', type=int, default=10000)
    args = parser.parse_args()
    if min(args.steps, args.per_class, args.batch_size, args.limit) < 1:
        parser.error('counts must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error('use a new empty output directory to avoid mixing results')
    out.mkdir(parents=True, exist_ok=True)
    model, config, prior, step = load_model(args.checkpoint, args.device)
    conditional = config['kind'].startswith('label')
    report = dict(vars(args), checkpoint_sha256=sha256(args.checkpoint),
                  checkpoint_step=step, kind=config['kind'], ema=True)
    if args.reverse_data_root:
        if prior is None:
            parser.error('label-free reverse classification requires a Gaussian model')
        data = CIFAR10(args.reverse_data_root, train=False, download=False)
        count = min(args.limit, len(data))
        predictions = []
        for start in range(0, count, args.batch_size):
            images = torch.from_numpy(data.data[start:min(count, start+args.batch_size)])
            images = images.permute(0, 3, 1, 2).to(args.device).float()/127.5-1
            encoded, _ = integrate(model, images, None, args.method, args.steps, reverse=True)
            if not torch.isfinite(encoded).all():
                raise RuntimeError('nonfinite reverse endpoint')
            predictions.extend(prior.classify(encoded).cpu().tolist())
        truth = np.asarray(data.targets[:count])
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (truth, predictions), 1)
        report.update(reverse_images=count, reverse_accuracy=float((truth == predictions).mean()),
                      predictions=predictions, labels=truth.tolist(), confusion_matrix=confusion.tolist())
    else:
        labels = torch.arange(10, device=args.device).repeat_interleave(args.per_class)
        rng = torch.Generator(device=args.device).manual_seed(args.seed)
        noise = initial_noise(labels, config, prior, rng)
        results = []
        for start in range(0, len(labels), args.batch_size):
            stop = start+args.batch_size
            generated, _ = integrate(model, noise[start:stop], labels[start:stop] if conditional else None,
                                     args.method, args.steps)
            if not torch.isfinite(generated).all():
                raise RuntimeError('nonfinite generated image')
            results.append(to_uint8(generated).cpu())
        images = torch.cat(results)
        save_image(images.float()/255, out/'generated-by-class.png', nrow=args.per_class)
        np.savez_compressed(out/'samples.npz', images=images.numpy(), labels=labels.cpu().numpy())
        report['generated_images'] = len(labels)
    write_json(out/'result.json', report)
    print('Wrote ' + str(out/'result.json'))


if __name__ == '__main__':
    main()
