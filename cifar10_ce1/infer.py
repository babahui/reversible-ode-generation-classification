"""Generate class grids or classify local images using the released EMA model."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

from .common import initial_noise, integrate, load_model, to_uint8, write_json, sha256

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='task', required=True)
    for task in ['sample', 'classify']:
        p = sub.add_parser(task)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--method', choices=['euler', 'heun'], default='heun' if task == 'sample' else 'euler')
        p.add_argument('--steps', type=int, default=60 if task == 'sample' else 1)
        p.add_argument('--batch-size', type=int, default=64)
        if task == 'sample':
            p.add_argument('--per-class', type=int, default=5)
            p.add_argument('--seed', type=int, default=20260912)
        else:
            p.add_argument('--images', nargs='+', required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or (args.task == 'sample' and args.per_class < 1):
        parser.error('Counts and steps must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config, prior, step = load_model(args.checkpoint, args.device)
    if prior is None:
        raise ValueError('Expected a CIFAR-10 Gaussian-prior checkpoint')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(vars(args), checkpoint_step=step, checkpoint_sha256=sha256(args.checkpoint),
                    weights='EMA', label_input=False)
    write_json(out/'protocol.json', protocol)
    with torch.inference_mode():
        if args.task == 'sample':
            labels = torch.arange(10, device=args.device).repeat_interleave(args.per_class)
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            noise = initial_noise(labels, config, prior, generator)
            images = []
            for batch in noise.split(args.batch_size):
                result, _ = integrate(model, batch, None, args.method, args.steps)
                images.append(to_uint8(result).cpu())
            images = torch.cat(images)
            save_image(images.float()/255, out/'grid.png', nrow=args.per_class)
            np.savez_compressed(out/'samples.npz', images=images.numpy(), labels=labels.cpu().numpy())
        else:
            predictions = []
            for start in range(0, len(args.images), args.batch_size):
                paths = args.images[start:start+args.batch_size]
                arrays = []
                for path in paths:
                    with Image.open(path) as image:
                        arrays.append(np.array(image.convert('RGB').resize((32,32), Image.Resampling.BILINEAR)))
                batch = torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).to(args.device).float()/127.5-1
                latent, _ = integrate(model, batch, None, args.method, args.steps, reverse=True)
                labels = prior.log_posterior(latent).argmax(1).cpu().tolist()
                predictions.extend(dict(image=path, class_id=label, class_name=CLASSES[label])
                                   for path, label in zip(paths, labels))
            write_json(out/'predictions.json', predictions)
            for prediction in predictions:
                print(prediction)


if __name__ == '__main__':
    main()
